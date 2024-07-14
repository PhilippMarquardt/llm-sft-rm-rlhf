import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class RewardModel(torch.nn.Module):
    def __init__(self, model_name):
        super(RewardModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

class RewardDataset:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        assert len(self.dataset["chosen"]) == len(self.dataset["rejected"])

    def __len__(self):
        return len(self.dataset["chosen"])

    def __getitem__(self, idx):
        chosen = self.dataset["chosen"][idx]
        rejected = self.dataset["rejected"][idx]
        c = self.tokenizer(chosen, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        r = self.tokenizer(rejected, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        return {
            "chosen_input_ids": c["input_ids"].squeeze(0),
            "chosen_attention_mask": c["attention_mask"].squeeze(0),
            "rejected_input_ids": r["input_ids"].squeeze(0),
            "rejected_attention_mask": r["attention_mask"].squeeze(0)
        }

def calculate_loss(chosen_scores, rejected_scores):
    """
    Calculate the loss such that chosen scores are higher than rejected scores.
    """
    return -nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()


def train_reward_model(model_name, num_epochs=15, batch_size=4, learning_rate=1e-4):
    ds = load_dataset("Anthropic/hh-rlhf")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    

    train_dataset = RewardDataset(tokenizer, ds["train"])
    eval_dataset = RewardDataset(tokenizer, ds["test"])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    model = RewardModel(model_name).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_eval_loss = float('inf')
    no_improvement = 0
    early_stopping_patience = 3
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            chosen_scores = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask).squeeze(-1)
            rejected_scores = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask).squeeze(-1)
    
            loss = calculate_loss(chosen_scores, rejected_scores)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    
            progress_bar.set_postfix({"Avg Loss": total_loss / (progress_bar.n + 1)})
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
    
        model.eval()
        eval_loss = 0
        eval_steps = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    
            with torch.no_grad():
                chosen_scores = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask).squeeze(-1)
                rejected_scores = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask).squeeze(-1)
                loss = calculate_loss(chosen_scores, rejected_scores)
    
            eval_loss += loss.item()
            eval_steps += 1
        eval_loss = eval_loss / eval_steps
        print(f"Evaluation loss: {eval_loss}")
    
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            no_improvement = 0
            model.model.save_pretrained("best_reward_model")
            tokenizer.save_pretrained("best_reward_model")
            print("New best model saved!")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    model.model.save_pretrained("final_reward_model")
    tokenizer.save_pretrained("final_reward_model")
    
    print("Training completed!")

if __name__ == "__main__":
    train_reward_model("roberta-base")