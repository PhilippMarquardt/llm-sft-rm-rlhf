import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pickle
import math
from torch.nn.utils import clip_grad_norm_

if __name__ == "__main__":

    ds = load_dataset("Anthropic/hh-rlhf")

    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    human_token = ""
    assistant_token = ""

    def tokenize_function(examples):
        conversations = examples["chosen"]
        tokenized_conversations = []
        
        for conversation in conversations:
            # # Split the conversation into turns
            # turns = conversation.split("\n\n")
            # formatted_conversation = ""
            
            # for turn in turns:
            #     if turn.startswith("Human:"):
            #         formatted_conversation += f"{turn[7:].strip()} "
            #     elif turn.startswith("Assistant:"):
            #         formatted_conversation += f"{turn[11:].strip()} "
            
            tokenized_conversations.append(conversation.replace("\n\n", ""))
        
        # Tokenize the formatted conversations
        inputs = tokenizer(tokenized_conversations, truncation=True, max_length=512, padding="max_length")
        inputs["labels"] = inputs["input_ids"].copy()
        
        # Convert lists to tensors
        inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"])
        inputs["labels"] = torch.tensor(inputs["labels"])
        
        return inputs
    
    pickle_path = "tokenized_datasettfinal.pkl"
    if os.path.exists(pickle_path):
        print("Loading tokenized dataset from pickle file...")
        with open(pickle_path, "rb") as f:
            tokenized_datasets = pickle.load(f)
    else:
        print("Tokenizing dataset...")
        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)
        print("Saving tokenized dataset to pickle file...")
        with open(pickle_path, "wb") as f:
            pickle.dump(tokenized_datasets, f)
    
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=4)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )

    def compute_perplexity(model, dataloader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = torch.stack(batch["input_ids"]).to(device)
                attention_mask = torch.stack(batch["attention_mask"]).to(device)
                labels = torch.stack(batch["labels"]).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item() * input_ids.size(0)
                total_tokens += attention_mask.sum().item()
        return math.exp(total_loss / total_tokens)
    
    def generate_sample(model, tokenizer, device, prompt=" Hi, how are you?"):
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, num_return_sequences=1, 
                                    no_repeat_ngram_size=2, top_k=50, top_p=0.95, 
                                    temperature=0.7, do_sample=True)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    model.train()
    best_eval_loss = float('inf')
    patience = 3
    no_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        
        for batch in progress_bar:

            input_ids = torch.stack(batch["input_ids"]).to(device)
            attention_mask = torch.stack(batch["attention_mask"]).to(device)
            labels = torch.stack(batch["labels"]).to(device)
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"Avg Loss": total_loss / (progress_bar.n + 1)})
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")

        model.eval()
        eval_loss = 0
        eval_steps = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = torch.stack(batch["input_ids"]).to(device)
            attention_mask = torch.stack(batch["attention_mask"]).to(device)
            labels = torch.stack(batch["labels"]).to(device)
    
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()
            eval_steps += 1
        eval_loss = eval_loss / eval_steps
        perplexity = compute_perplexity(model, eval_dataloader, device)
        print(f"Evaluation loss: {eval_loss}")
        print(f"Perplexity: {perplexity}")

        sample_text = generate_sample(model, tokenizer, device)
        print("Generated sample:")
        print(sample_text)
        print("-" * 50)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            no_improvement = 0

            model.save_pretrained("./best_fine_tuned_gpt2")
            tokenizer.save_pretrained("./best_fine_tuned_gpt2")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
        model.train()

    model.save_pretrained("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")
    
    print("Training completed!")
