#!/usr/bin/python3
"""
Training script for Question Generation Agent (Q-Agent)
Supports: LoRA fine-tuning, Full fine-tuning, and Prompt tuning
Optimized for AMD MI300 GPU with 192GB HBM
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


@dataclass
class TrainingConfig:
    """Configuration for training the Q-Agent"""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./models/q-agent-finetuned"
    
    # Training data
    train_data_path: str = "./data/question_training_data.json"
    val_data_path: Optional[str] = None
    
    # Training mode: "lora", "full", "prompt"
    training_mode: str = "lora"
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    
    # Optimization
    optim: str = "adamw_torch"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True  # MI300 supports bfloat16
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Advanced settings
    max_grad_norm: float = 1.0
    seed: int = 42


class QuestionDatasetBuilder:
    """Build training datasets for question generation"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # System prompt for question generation
        self.system_prompt = """You are an expert question generator specializing in creating challenging, valid multiple-choice questions."""
    
    def create_training_prompt(self, example: Dict) -> str:
        """Create a training prompt from a question example"""
        
        # Input: topic and difficulty
        topic = example.get("topic", "General Knowledge")
        difficulty = example.get("difficulty", "medium")
        
        # Output: the complete question in JSON format
        output_json = {
            "topic": topic,
            "difficulty": difficulty,
            "question": example["question"],
            "choices": example["choices"],
            "answer": example["answer"],
            "explanation": example.get("explanation", "")
        }
        
        user_message = f"Generate a {difficulty} difficulty multiple-choice question on the topic: {topic}"
        assistant_message = json.dumps(output_json, indent=2)
        
        # Format as chat
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def load_and_prepare_data(self, data_path: str) -> Dataset:
        """Load and prepare training data"""
        
        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        # Create formatted prompts
        formatted_data = []
        for example in data:
            try:
                prompt = self.create_training_prompt(example)
                formatted_data.append({"text": prompt})
            except Exception as e:
                print(f"Warning: Skipping example due to error: {e}")
                continue
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)
        return dataset


class QAgentTrainer:
    """Trainer for Question Generation Agent"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        
    def setup_environment(self):
        """Setup training environment"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.manual_seed(self.config.seed)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        
        print(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"  # Required for training
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        return model, tokenizer
    
    def prepare_model_for_training(self, model):
        """Prepare model based on training mode"""
        
        if self.config.training_mode == "lora":
            print("Configuring LoRA fine-tuning...")
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        elif self.config.training_mode == "full":
            print("Configuring full fine-tuning...")
            # All parameters are trainable by default
            
        elif self.config.training_mode == "prompt":
            print("Configuring prompt tuning...")
            # Implement prompt tuning if needed
            raise NotImplementedError("Prompt tuning not yet implemented")
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""
        
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            optim=self.config.optim,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=self.config.max_grad_norm,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=["tensorboard"],
            save_strategy="steps",
            evaluation_strategy="steps" if self.config.val_data_path else "no",
            load_best_model_at_end=True if self.config.val_data_path else False,
            seed=self.config.seed,
        )
    
    def train(self):
        """Main training loop"""
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare model for training
        model = self.prepare_model_for_training(model)
        
        # Load and prepare datasets
        dataset_builder = QuestionDatasetBuilder(tokenizer, self.config.max_seq_length)
        train_dataset = dataset_builder.load_and_prepare_data(self.config.train_data_path)
        
        eval_dataset = None
        if self.config.val_data_path:
            eval_dataset = dataset_builder.load_and_prepare_data(self.config.val_data_path)
        
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Validation samples: {len(eval_dataset)}")
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,  # Don't pack sequences for question generation
        )
        
        # Train
        print("\n" + "="*70)
        print("Starting training...")
        print("="*70 + "\n")
        
        trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"\n✅ Training complete! Model saved to: {self.config.output_dir}")
        
        return trainer


def create_sample_training_data(output_path: str = "./data/question_training_data.json"):
    """Create sample training data for demonstration"""
    
    sample_questions = [
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
            "explanation": "The sequence follows the pattern n(n+1) where n starts from 2. So 2×3=6, 3×4=12, 4×5=20, 5×6=30, 6×7=42."
        },
        {
            "topic": "Probability",
            "difficulty": "medium",
            "question": "A fair die is rolled twice. What is the probability that the sum is 7?",
            "choices": [
                "A) 1/6",
                "B) 1/12",
                "C) 1/9",
                "D) 1/8"
            ],
            "answer": "A",
            "explanation": "There are 6 ways to get a sum of 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) out of 36 total outcomes. 6/36 = 1/6."
        },
        {
            "topic": "Data Structures",
            "difficulty": "hard",
            "question": "What is the time complexity of finding the kth smallest element in an unsorted array using QuickSelect algorithm on average?",
            "choices": [
                "A) O(n log n)",
                "B) O(n²)",
                "C) O(n)",
                "D) O(k log n)"
            ],
            "answer": "C",
            "explanation": "QuickSelect has an average time complexity of O(n) as it only recurses into one partition, unlike QuickSort which recurses into both."
        }
    ]
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample training data at: {output_path}")


if __name__ == "__main__":
    # Create sample training data
    create_sample_training_data()
    
    # Configure training
    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="./models/q-agent-lora",
        train_data_path="./data/question_training_data.json",
        training_mode="lora",  # Use LoRA for efficiency
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=2048,
        bf16=True,  # Use bfloat16 for MI300
    )
    
    # Initialize trainer
    trainer = QAgentTrainer(config)
    
    # Start training
    trainer.train()
