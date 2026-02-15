# AMD AI Premier League (AAIPL) - Complete Solution

## 🎯 Competition Overview

A head-to-head AI competition where teams build two intelligent agents:
- **Q-Agent**: Generates valid, challenging multiple-choice questions
- **A-Agent**: Answers questions from opposing teams

**Goal**: Create questions that stump opponents while answering their questions correctly.

## 📊 Scoring System

### Formulas

```
A-Agent Score = (Questions Correctly Answered / N) × 100
Q-Agent Score = (Questions Incorrectly Answered by Opponent / N) × 100
Total Score = Q-Agent Score + A-Agent Score
```

Where `N` = number of format-correct questions

### Disqualification Rules

- Teams with **<50% format-correct questions** are automatically disqualified
- `num_questions` ranges from 2 to 1000+

### Tiebreaker

In case of a tie, closed benchmark questions evaluate A-agents to rank teams.

## 📁 Project Structure

```
AMD/
├── Core Agents
│   ├── question_agent.py          # Question generation agent
│   ├── answer_agent.py            # Answer generation agent
│   ├── question_model.py          # Qwen-based Q model
│   ├── question_model_llama.py    # Llama-based Q model
│   ├── answer_model.py            # Qwen-based A model
│   └── answer_model_llama.py      # Llama-based A model
│
├── Training Scripts
│   ├── train_question_agent.py    # SFT for Q-Agent
│   ├── train_answer_agent.py      # SFT for A-Agent
│   └── train_rl.py               # DPO reinforcement learning
│
├── Evaluation & Competition
│   ├── evaluator.py              # Scoring and validation
│   └── match_orchestrator.py     # Match/tournament runner
│
├── Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── TRAINING_GUIDE.md        # Training instructions
│   └── README.md                # This file
│
└── Data & Results
    ├── data/                     # Training data
    ├── models/                   # Fine-tuned models
    └── match_results/            # Competition results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install ROCm-compatible PyTorch for AMD MI300 GPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Agents (Default Models)

#### Test Q-Agent
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

#### Test A-Agent
```python
from answer_agent import AnsweringAgent

a_agent = AnsweringAgent()
answer = a_agent.answer_question({
    "question": "What is the capital of France?",
    "choices": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"]
})
print(answer)
```

### 3. Run a Match

```python
from match_orchestrator import MatchOrchestrator, TeamConfig

# Define teams
team_a = TeamConfig(name="Team Alpha", use_default_models=True)
team_b = TeamConfig(name="Team Beta", use_default_models=True)

# Create orchestrator
orchestrator = MatchOrchestrator(
    num_questions=10,
    difficulty="medium",
    min_valid_pct=50.0
)

# Run match
result = orchestrator.run_match(team_a, team_b)
```

## 🎓 Training Your Agents

### Phase 1: Supervised Fine-Tuning (SFT)

#### Train Q-Agent
```bash
python train_question_agent.py
```

#### Train A-Agent
```bash
python train_answer_agent.py
```

**Configuration** (edit in training scripts):
```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./models/q-agent-lora",
    training_mode="lora",  # or "full"
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    bf16=True,  # For MI300
)
```

### Phase 2: Reinforcement Learning (DPO)

```bash
python train_rl.py
```

**How it works**:
1. Generate multiple question/answer candidates
2. Score them based on quality metrics
3. Create preference pairs (good vs bad)
4. Train with DPO to prefer high-quality outputs

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

## 📝 Data Formats

### Question Format (Q-Agent Output)

```json
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
```

### Answer Format (A-Agent Output)

```json
{
  "answer": "B",
  "confidence": 0.95,
  "reasoning": "The pattern is n(n+1), so next is 6×7=42"
}
```

## 🏆 Competition Strategy

### 1. Question Generation Strategy

**Make questions challenging but fair**:
- Use edge cases and tricky scenarios
- Create plausible distractors (wrong answers that seem reasonable)
- Cover diverse topics to maximize opponent's knowledge gaps
- Balance difficulty: too easy = low Q-score, too hard = might be invalid

**Example good question**:
```
"What is the time complexity of QuickSelect on average?"
Choices:
A) O(n log n)  ← Looks like QuickSort
B) O(n²)       ← Worst case
C) O(n)        ← CORRECT
D) O(k log n)  ← Plausible but wrong
```

### 2. Answer Generation Strategy

**Maximize accuracy**:
- Use lower temperature (0.3-0.5) for consistent answers
- Train on diverse question types
- Implement reasoning chains
- Validate answer format before submission

### 3. Training Data Strategy

**Quality > Quantity**:
- Collect 500-1000 high-quality examples per agent
- Cover multiple domains: Math, Science, Logic, CS, etc.
- Include various difficulty levels
- Use data augmentation to expand dataset

### 4. 24-Hour Timeline (MI300 GPU)

| Hours | Activity |
|-------|----------|
| 0-2   | Data collection & preparation |
| 2-6   | Train Q-Agent (LoRA) |
| 6-10  | Train A-Agent (LoRA) |
| 10-14 | Test & iterate, generate preference data |
| 14-18 | DPO training for both agents |
| 18-22 | Final testing & optimization |
| 22-24 | Prepare submission & documentation |

## 🔧 Evaluation System

### Validation Rules

**Questions must have**:
- ✅ Valid JSON format
- ✅ Required fields: topic, question, choices, answer, explanation
- ✅ Exactly 4 choices (A, B, C, D)
- ✅ Question ends with `?`
- ✅ Answer is one of A/B/C/D
- ✅ Choices formatted as "A) ...", "B) ...", etc.

**Answers must have**:
- ✅ Valid JSON format
- ✅ Required field: answer (A/B/C/D)
- ✅ Optional: confidence (0.0-1.0), reasoning

### Running Evaluation

```python
from evaluator import MatchEvaluator

evaluator = MatchEvaluator(min_valid_pct=50.0)

result = evaluator.evaluate_match(
    team_a_name="Team Alpha",
    team_b_name="Team Beta",
    team_a_questions=team_a_questions,
    team_b_questions=team_b_questions,
    team_a_answers=team_a_answers,
    team_b_answers=team_b_answers
)

evaluator.print_match_summary(result)
```

## 📈 Monitoring & Debugging

### TensorBoard

```bash
tensorboard --logdir ./models/q-agent-lora/logs
```

### Check GPU Utilization

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
```

### Common Issues

**Out of Memory**:
```python
# Reduce batch size
per_device_train_batch_size=2
gradient_accumulation_steps=8

# Enable optimizations
gradient_checkpointing=True
bf16=True
```

**Poor Question Quality**:
- Increase training data diversity
- Use DPO to filter bad examples
- Adjust temperature (0.7-0.9 for creativity)
- Add more examples of good questions

**Low Answer Accuracy**:
- Train on more diverse questions
- Use lower temperature (0.3-0.5)
- Add reasoning in training data
- Fine-tune on benchmark questions

## 🎯 Advanced Techniques

### 1. Adversarial Training
```python
# Train Q-agent to generate questions that fool your own A-agent
# Then train A-agent on those hard questions
# Iterate to improve both agents
```

### 2. Ensemble Methods
```python
# Use multiple model checkpoints
# Aggregate predictions for higher accuracy
```

### 3. Domain-Specific Fine-tuning
```python
# Fine-tune separate models for different domains
# Route questions to appropriate specialist
```

### 4. Active Learning
```python
# Identify weak areas from validation
# Generate more training data for those areas
# Retrain to improve
```

## 📊 Example Match Output

```
================================================================================
MATCH RESULT: Team Alpha vs Team Beta
================================================================================

Team Alpha Performance:
  Q-Agent:
    - Questions Generated: 10
    - Valid Questions: 9 (90.0%)
    - Questions Answered Incorrectly by Opponent: 4
    - Q-Agent Score: 44.44
  A-Agent:
    - Answers Submitted: 8
    - Valid Answers: 8
    - Correct Answers: 6
    - A-Agent Score: 75.00
  TOTAL SCORE: 119.44

Team Beta Performance:
  Q-Agent:
    - Questions Generated: 10
    - Valid Questions: 8 (80.0%)
    - Questions Answered Incorrectly by Opponent: 3
    - Q-Agent Score: 37.50
  A-Agent:
    - Answers Submitted: 9
    - Valid Answers: 9
    - Correct Answers: 5
    - A-Agent Score: 55.56
  TOTAL SCORE: 93.06

================================================================================
🏆 WINNER: Team Alpha
================================================================================
```

## 📚 Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Guide](https://huggingface.co/docs/peft)
- [TRL (RL Training)](https://huggingface.co/docs/trl)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

## 🎬 Deliverables Checklist

- [ ] Working Q-Agent generating valid MCQs
- [ ] Working A-Agent answering questions accurately
- [ ] Training scripts and configurations
- [ ] Evaluation results showing performance
- [ ] Slides/video explaining techniques used
- [ ] Model checkpoints (if submitting)
- [ ] Documentation of approach

## 💡 Tips for Success

1. **Start Simple**: Test with default models first
2. **Validate Early**: Check question/answer formats constantly
3. **Diverse Data**: Cover multiple topics and difficulty levels
4. **Iterate Fast**: Use LoRA for quick experiments
5. **Monitor Quality**: Track validation metrics during training
6. **Test Adversarially**: Have your agents compete during development
7. **Optimize for Rules**: Remember the 50% validity threshold
8. **Document Everything**: Keep notes for your presentation

## 🏅 Good Luck!

You now have a complete system for the AMD AI Premier League. Focus on:
- **Data quality** over quantity
- **Validation** to avoid disqualification
- **Adversarial testing** to find weaknesses
- **Iterative improvement** using RL

**May the best agents win! 🚀**

---

## 📧 Support

For questions about the code, refer to:
- Individual file docstrings
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Competition guidelines from AMD
