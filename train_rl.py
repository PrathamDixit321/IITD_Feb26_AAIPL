#!/usr/bin/python3
"""
Reinforcement Learning Training for Q-Agent and A-Agent
Uses DPO (Direct Preference Optimization) for iterative improvement
Optimized for AMD MI300 GPU
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer


@dataclass
class RLConfig:
    """Configuration for RL training"""
    
    # Model settings
    model_name: str = "./models/q-agent-lora"  # Path to fine-tuned model
    output_dir: str = "./models/q-agent-rl"
    
    # Training data
    preference_data_path: str = "./data/preference_data.json"
    
    # DPO settings
    beta: float = 0.1  # KL penalty coefficient
    
    # LoRA settings (if continuing with LoRA)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training hyperparameters
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    
    # Optimization
    optim: str = "adamw_torch"
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
    
    seed: int = 42


class PreferenceDataBuilder:
    """Build preference datasets for DPO training"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def create_preference_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> Dict:
        """Create a preference pair for DPO"""
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    
    def generate_preference_data_from_qa(
        self,
        q_agent,
        a_agent,
        topics: List[str],
        num_samples_per_topic: int = 5
    ) -> List[Dict]:
        """
        Generate preference data by:
        1. Q-agent generates multiple questions
        2. A-agent answers them
        3. Score answers to create preference pairs
        """
        
        preference_data = []
        
        for topic in topics:
            print(f"Generating preference data for topic: {topic}")
            
            # Generate multiple questions for this topic
            for difficulty in ["easy", "medium", "hard"]:
                # Q-agent generates a question
                question_prompt = f"Generate a {difficulty} MCQ question on: {topic}"
                
                # Generate multiple candidate questions
                questions = []
                for _ in range(num_samples_per_topic):
                    response, _, _ = q_agent.generate_response(
                        question_prompt,
                        max_new_tokens=512,
                        temperature=0.8,
                        do_sample=True
                    )
                    questions.append(response)
                
                # Score questions based on quality metrics
                # (In practice, you'd have a more sophisticated scoring function)
                scored_questions = self._score_questions(questions)
                
                # Create preference pairs: best vs worst
                if len(scored_questions) >= 2:
                    best = scored_questions[0][0]
                    worst = scored_questions[-1][0]
                    
                    preference_data.append({
                        "prompt": question_prompt,
                        "chosen": best,
                        "rejected": worst
                    })
        
        return preference_data
    
    def _score_questions(self, questions: List[str]) -> List[Tuple[str, float]]:
        """
        Score questions based on quality metrics
        Returns: List of (question, score) tuples sorted by score (descending)
        """
        
        scored = []
        for q in questions:
            score = 0.0
            
            # Simple heuristics (replace with better metrics)
            # 1. Has proper JSON structure
            try:
                q_json = json.loads(q)
                score += 2.0
                
                # 2. Has all required fields
                if all(k in q_json for k in ["question", "choices", "answer", "explanation"]):
                    score += 2.0
                
                # 3. Question ends with ?
                if q_json.get("question", "").strip().endswith("?"):
                    score += 1.0
                
                # 4. Has 4 choices
                if len(q_json.get("choices", [])) == 4:
                    score += 1.0
                
                # 5. Answer is valid (A/B/C/D)
                if q_json.get("answer") in ["A", "B", "C", "D"]:
                    score += 1.0
                
                # 6. Has explanation
                if len(q_json.get("explanation", "")) > 20:
                    score += 1.0
                
            except:
                score = 0.0
            
            scored.append((q, score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def load_preference_data(self, data_path: str) -> Dataset:
        """Load preference data from JSON file"""
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Dataset.from_list(data)
    
    def save_preference_data(self, data: List[Dict], output_path: str):
        """Save preference data to JSON file"""
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} preference pairs to {output_path}")


class RLTrainer:
    """Reinforcement Learning Trainer using DPO"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.setup_environment()
    
    def setup_environment(self):
        """Setup training environment"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.manual_seed(self.config.seed)
        
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        
        print(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load reference model (for DPO)
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        return model, ref_model, tokenizer
    
    def train(self):
        """Main DPO training loop"""
        
        # Load models
        model, ref_model, tokenizer = self.load_model_and_tokenizer()
        
        # Load preference dataset
        dataset_builder = PreferenceDataBuilder(tokenizer)
        train_dataset = dataset_builder.load_preference_data(self.config.preference_data_path)
        
        print(f"Training samples: {len(train_dataset)}")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            optim=self.config.optim,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=["tensorboard"],
            seed=self.config.seed,
        )
        
        # Create DPO trainer
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=self.config.beta,
            max_prompt_length=self.config.max_prompt_length,
            max_length=self.config.max_seq_length,
        )
        
        # Train
        print("\n" + "="*70)
        print("Starting DPO training...")
        print("="*70 + "\n")
        
        dpo_trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        dpo_trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"\n✅ DPO training complete! Model saved to: {self.config.output_dir}")
        
        return dpo_trainer


def create_sample_preference_data(output_path: str = "./data/preference_data.json"):
    """Create sample preference data for demonstration"""
    
    sample_preferences = [
        {
            "prompt": "Generate a hard MCQ question on: Number Series",
            "chosen": json.dumps({
                "topic": "Number Series",
                "difficulty": "hard",
                "question": "What is the next number in the sequence: 1, 4, 9, 16, 25, ?",
                "choices": ["A) 30", "B) 36", "C) 40", "D) 49"],
                "answer": "B",
                "explanation": "This is a sequence of perfect squares: 1², 2², 3², 4², 5², so next is 6² = 36."
            }, indent=2),
            "rejected": json.dumps({
                "topic": "Number Series",
                "question": "What comes next: 1, 4, 9, 16, 25",  # Missing question mark
                "choices": ["A) 30", "B) 36"],  # Only 2 choices
                "answer": "B"
                # Missing explanation
            }, indent=2)
        },
        {
            "prompt": "Generate a medium MCQ question on: Probability",
            "chosen": json.dumps({
                "topic": "Probability",
                "difficulty": "medium",
                "question": "What is the probability of getting exactly 2 heads in 3 coin flips?",
                "choices": ["A) 1/8", "B) 1/4", "C) 3/8", "D) 1/2"],
                "answer": "C",
                "explanation": "There are 3 ways to get exactly 2 heads (HHT, HTH, THH) out of 8 total outcomes. Probability = 3/8."
            }, indent=2),
            "rejected": json.dumps({
                "topic": "Probability",
                "question": "Coin flip probability",  # Vague question
                "choices": ["A) Yes", "B) No", "C) Maybe", "D) Sometimes"],  # Nonsensical choices
                "answer": "A",
                "explanation": "Coins have probability."  # Poor explanation
            }, indent=2)
        }
    ]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_preferences, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample preference data at: {output_path}")


if __name__ == "__main__":
    # Create sample preference data
    create_sample_preference_data()
    
    # Configure RL training
    config = RLConfig(
        model_name="./models/q-agent-lora",  # Use your fine-tuned model
        output_dir="./models/q-agent-rl",
        preference_data_path="./data/preference_data.json",
        beta=0.1,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        bf16=True,
    )
    
    # Initialize RL trainer
    trainer = RLTrainer(config)
    
    # Start DPO training
    trainer.train()
