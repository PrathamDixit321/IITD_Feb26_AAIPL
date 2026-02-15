# AMD AI Premier League (AAIPL) - Training Guide

## 🎯 Overview

This repository contains training scripts for the **Q-Agent** (Question Generator) and **A-Agent** (Answer Agent) for the AMD AI Premier League competition.

## 📁 Project Structure

```
AMD/
├── question_agent.py          # Question generation agent
├── answer_agent.py            # Answer generation agent
├── question_model.py          # Qwen-based Q model
├── question_model_llama.py    # Llama-based Q model
├── answer_model.py            # Qwen-based A model
├── answer_model_llama.py      # Llama-based A model
├── train_question_agent.py    # Training script for Q-Agent
├── train_answer_agent.py      # Training script for A-Agent
├── train_rl.py               # Reinforcement learning (DPO)
├── requirements.txt          # Python dependencies
└── data/                     # Training data directory
    ├── question_training_data.json
    ├── answer_training_data.json
    └── preference_data.json
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install ROCm-compatible PyTorch for AMD MI300 GPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify GPU Access

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected output for MI300:
```
GPU Available: True
GPU Name: AMD Instinct MI300X
GPU Memory: 192.00 GB
```

## 📚 Training Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)

#### Train Q-Agent (Question Generator)

```bash
# Using LoRA (recommended for efficiency)
python train_question_agent.py
```

**Configuration options** (edit in `train_question_agent.py`):
```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./models/q-agent-lora",
    train_data_path="./data/question_training_data.json",
    training_mode="lora",  # or "full" for full fine-tuning
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,  # Use bfloat16 for MI300
)
```

#### Train A-Agent (Answer Agent)

```bash
python train_answer_agent.py
```

**Configuration options**:
```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./models/a-agent-lora",
    train_data_path="./data/answer_training_data.json",
    training_mode="lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
)
```

### Phase 2: Reinforcement Learning (Optional but Recommended)

Use **DPO (Direct Preference Optimization)** to improve agent quality:

```bash
python train_rl.py
```

**How it works**:
1. Generate multiple question/answer candidates
2. Score them based on quality metrics
3. Create preference pairs (good vs bad examples)
4. Train with DPO to prefer high-quality outputs

## 📊 Data Format

### Question Training Data (`question_training_data.json`)

```json
[
  {
    "topic": "Number Series",
    "difficulty": "hard",
    "question": "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
    "choices": [
      "A) 40",
      "B) 42",
      "C) 44",
      "D) 48"
    ],
    "answer": "B",
    "explanation": "The sequence follows n(n+1): 2×3=6, 3×4=12, ..., 6×7=42"
  }
]
```

### Answer Training Data (`answer_training_data.json`)

```json
[
  {
    "question": "What is the capital of France?",
    "choices": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
    "answer": "B",
    "confidence": 0.98,
    "reasoning": "Paris is the capital and largest city of France."
  }
]
```

### Preference Data (`preference_data.json`)

```json
[
  {
    "prompt": "Generate a hard MCQ on Number Series",
    "chosen": "{\"question\": \"...\", \"choices\": [...], ...}",
    "rejected": "{\"question\": \"bad question\", ...}"
  }
]
```

## ⚙️ Training Strategies

### Strategy 1: LoRA Fine-Tuning (Recommended)

**Pros**:
- Memory efficient (can train larger models)
- Fast training
- Easy to merge/switch adapters
- ~10-20% of full model parameters

**Cons**:
- Slightly lower performance than full fine-tuning

**Best for**: Limited time, multiple experiments

### Strategy 2: Full Fine-Tuning

**Pros**:
- Maximum performance
- Full model adaptation

**Cons**:
- Requires more memory
- Slower training
- Higher risk of overfitting

**Best for**: Final model after hyperparameter tuning

### Strategy 3: Prompt Tuning

**Pros**:
- Extremely parameter efficient
- Very fast

**Cons**:
- Limited adaptation capability

**Best for**: Quick baselines

## 🎮 Training on MI300 GPU (24 hours)

### Recommended Schedule

**Hours 0-8: Data Preparation & Initial Training**
- Collect/generate training data
- Train Q-Agent with LoRA (3-4 hours)
- Train A-Agent with LoRA (3-4 hours)

**Hours 8-16: Evaluation & Iteration**
- Test agents against each other
- Identify weaknesses
- Generate preference data
- Start DPO training

**Hours 16-22: Reinforcement Learning**
- DPO training for Q-Agent
- DPO training for A-Agent
- Fine-tune hyperparameters

**Hours 22-24: Final Testing & Optimization**
- Final evaluation
- Model merging/selection
- Prepare submission

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./models/q-agent-lora/logs
```

Open `http://localhost:6006` to view:
- Training loss
- Learning rate schedule
- Gradient norms
- GPU utilization

### Weights & Biases (Optional)

```python
# In training config
report_to=["wandb"]
```

## 🧪 Testing Your Agents

### Test Q-Agent

```python
from question_agent import QuestioningAgent

q_agent = QuestioningAgent()
question = q_agent.generate_question(
    topic="Machine Learning",
    difficulty="hard",
    max_new_tokens=512
)
print(question)
```

### Test A-Agent

```python
from answer_agent import AnsweringAgent

a_agent = AnsweringAgent()
answer = a_agent.answer_question({
    "question": "What is 2+2?",
    "choices": ["A) 3", "B) 4", "C) 5", "D) 6"]
})
print(answer)
```

## 🏆 Competition Tips

1. **Data Quality > Quantity**: Focus on high-quality, diverse training examples
2. **Difficulty Balance**: Train on easy/medium/hard questions
3. **Adversarial Training**: Have agents compete during training
4. **Domain Coverage**: Cover multiple topics (math, science, logic, etc.)
5. **Validation**: Always validate on held-out data
6. **Ensemble**: Consider using multiple models/checkpoints

## 🔧 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
per_device_train_batch_size=2

# Increase gradient accumulation
gradient_accumulation_steps=8

# Enable gradient checkpointing
gradient_checkpointing=True

# Use smaller model
model_name="Qwen/Qwen2.5-3B-Instruct"
```

### Slow Training

```python
# Use bfloat16
bf16=True

# Increase batch size (if memory allows)
per_device_train_batch_size=8

# Reduce logging
logging_steps=50
```

### Poor Performance

- Increase training data
- Increase epochs
- Adjust learning rate
- Try different models
- Use DPO/RL training

## 📝 Deliverables Checklist

- [ ] Working Q-Agent that generates valid MCQs
- [ ] Working A-Agent that answers questions accurately
- [ ] Training scripts and configurations
- [ ] Slides/video explaining techniques
- [ ] Model checkpoints
- [ ] Evaluation results

## 🎓 Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Guide](https://huggingface.co/docs/peft)
- [TRL (RL Training)](https://huggingface.co/docs/trl)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## 📧 Support

For questions or issues, refer to the competition guidelines or AMD documentation.

---

**Good luck! 🚀**
