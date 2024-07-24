import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

dataset = load_dataset("yitingxie/rlhf-reward-datasets")

access_token = ""
model_name = r"C:\Users\philmarq\source\repos\llm\gpt2mediumsft\checkpoint-28000"
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)  # Disable caching
model.gradient_checkpointing_enable()  # Enable gradient checkpointing
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", token=access_token)
tokenizer.pad_token = tokenizer.eos_token

ref_model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
ref_model.gradient_checkpointing_enable()

def truncate_input(example, max_length=512):
    prompt_ids = tokenizer.encode(example['prompt'], truncation=True, max_length=max_length//2)
    chosen_ids = tokenizer.encode(example['chosen'], truncation=True, max_length=max_length//2)
    rejected_ids = tokenizer.encode(example['rejected'], truncation=True, max_length=max_length//2)
    
    example['prompt'] = tokenizer.decode(prompt_ids)
    example['chosen'] = tokenizer.decode(chosen_ids)
    example['rejected'] = tokenizer.decode(rejected_ids)
    
    return example

truncated_dataset = dataset.map(truncate_input, fn_kwargs={'max_length': 512})

training_args = DPOConfig(
    output_dir="./results_dpo",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Increased for memory efficiency
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    remove_unused_columns=False,
    logging_steps=10,
    max_prompt_length=256,  # Explicitly set max_prompt_length
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    warmup_steps=100,
    max_length=512,
    beta=0.1,
    label_smoothing=0.0,
    loss_type="sigmoid",
    gradient_checkpointing=False,  # Enable gradient checkpointing
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=truncated_dataset["train"],
    eval_dataset=truncated_dataset["test"],
    tokenizer=tokenizer,
)

dpo_trainer.train()

dpo_trainer.save_model("./fine_tuned_model_dpo")
print("DPO Training completed!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_sample(model, tokenizer, device, prompt="Human: Hi, how are you?"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, 
                                no_repeat_ngram_size=2, top_k=50, top_p=0.95, 
                                temperature=0.7, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

sample_text = generate_sample(model, tokenizer, device)
print("Generated sample:")
print(sample_text)