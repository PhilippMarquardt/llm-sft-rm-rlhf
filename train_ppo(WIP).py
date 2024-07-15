import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Set CUDA device and disable wandb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Set CUDA device and disable wandb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

class LanguageModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate(self, input_ids, attention_mask, max_new_tokens):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)

class RewardModel(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.initial_model = LanguageModel(config.model_name).to(self.device)
        self.tuned_model = LanguageModel(config.model_name).to(self.device)
        self.reward_model = RewardModel(config.reward_model_name).to(self.device)
        
        self.optimizer = optim.Adam(self.tuned_model.parameters(), lr=config.lr)
        
        self.dataset = load_dataset("yitingxie/rlhf-reward-datasets", split="train")
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        self.writer = SummaryWriter(f'runs/ppo_rlhf_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    def collate_fn(self, batch):
        prompts = [item['prompt'] for item in batch]
        encoded_prompts = self.initial_model.tokenizer(prompts, return_tensors='pt', truncation=True, max_length=self.config.max_length, padding="max_length")
        return {
            'prompt_input_ids': encoded_prompts['input_ids'],
            'prompt_attention_mask': encoded_prompts['attention_mask'],
        }

    def compute_rewards(self, generated_texts):
        encoded = self.reward_model.tokenizer(generated_texts, return_tensors='pt', truncation=True, 
                                              max_length=self.config.max_length, padding="max_length")
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards.squeeze(-1)

    def compute_kl_divergence(self, tuned_logits, initial_logits):
        tuned_probs = torch.nn.functional.softmax(tuned_logits, dim=-1)
        initial_probs = torch.nn.functional.softmax(initial_logits, dim=-1)
        kl_div = torch.sum(tuned_probs * (torch.log(tuned_probs) - torch.log(initial_probs)), dim=-1)
        return kl_div

    def train(self):
        scaler = GradScaler()

        for epoch in range(self.config.num_epochs):
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                prompt_input_ids = batch['prompt_input_ids'].to(self.device)
                prompt_attention_mask = batch['prompt_attention_mask'].to(self.device)

                with torch.no_grad():
                    tuned_outputs = self.tuned_model.generate(prompt_input_ids, prompt_attention_mask, max_new_tokens=self.config.max_new_tokens)
                tuned_texts = self.tuned_model.tokenizer.batch_decode(tuned_outputs, skip_special_tokens=True)

                rewards = self.compute_rewards(tuned_texts)
                tuned_logits = self.tuned_model(tuned_outputs, attention_mask=None).logits
                with torch.no_grad():
                    initial_logits = self.initial_model(tuned_outputs, attention_mask=None).logits

                kl_div = self.compute_kl_divergence(tuned_logits, initial_logits)

                old_log_probs = torch.nn.functional.log_softmax(initial_logits, dim=-1)
                new_log_probs = torch.nn.functional.log_softmax(tuned_logits, dim=-1)

                old_log_probs = torch.gather(old_log_probs, 2, tuned_outputs.unsqueeze(-1)).squeeze(-1)
                new_log_probs = torch.gather(new_log_probs, 2, tuned_outputs.unsqueeze(-1)).squeeze(-1)

                ratio = torch.exp(new_log_probs - old_log_probs)

                rewards = rewards.unsqueeze(-1).expand_as(new_log_probs)
                kl_div = kl_div.expand_as(new_log_probs)
                advantage = rewards - self.config.kl_coef * kl_div

                ppo_loss = -torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
                ).mean()

                # Optimize
                self.optimizer.zero_grad()
                scaler.scale(ppo_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.writer.add_scalar('Loss/ppo', ppo_loss.item(), epoch)
                self.writer.add_scalar('KL_Divergence', kl_div.mean().item(), epoch)
                self.writer.add_scalar('Reward', rewards.mean().item(), epoch)

        self.writer.close()

class Config:
    def __init__(self):
        self.model_name = 'gpt2'
        self.reward_model_name = 'roberta-base'
        self.dataset_name = 'yitingxie/rlhf-reward-datasets'
        self.batch_size = 4
        self.num_epochs = 5
        self.max_length = 512
        self.max_new_tokens = 20
        self.lr = 1e-5
        self.clip_epsilon = 0.2
        self.kl_coef = 0.1

if __name__ == "__main__":
    config = Config()
    trainer = PPOTrainer(config)
    trainer.train()