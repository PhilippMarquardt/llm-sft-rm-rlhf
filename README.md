# llm-sft-rm-rlhf

This repository contains files to guide a large language model to generate human-likable content. This is generally done in 3 steps:

1. **Supervised Fine Tuning (SFT)**: Typically done by performing next token prediction on a curated dataset. This is similar to unsupervised pretraining.
2. **Reward Model Training**: A reward model is trained which is later used in Reinforcement Learning from Human Feedback (RLHF) to provide the reinforcement learning model its reward.
3. **Reinforcement Learning from Human Feedback (RLHF)**: The SFT fine-tuned model and the reward model are used together. The RLHF process optimizes the language model's outputs to maximize the reward, thereby generating more human-likable content.

All files can be simply run without any arguments. SFT will train a GPT-2 model fine-tuned on the [Anthropic/hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf). The reward model is trained on the same dataset but uses a RoBERTa model to leverage bidirectional attention.
