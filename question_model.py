
# Qwen/Qwen2.5-7B-Instruct based Q-Agent
import time
import torch
import re
from typing import Optional, Union, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class QAgent(object):
    def __init__(self, **kwargs):
        # Allow loading fine-tuned models or different base models
        self.model_name = kwargs.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        
        print(f"Loading Q-Agent model: {self.model_name}")
        
        # Determine device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use bfloat16 for MI300/Ampere+, float16 for others
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding_side="left",
                trust_remote_code=True
            )
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=self.dtype, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set evaluation mode
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Fallback: Creating dummy model for testing/no-gpu environment")
            self.model = None
            self.tokenizer = None

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> Tuple[Union[str, List[str]], Optional[int], Optional[float]]:
        
        # Mock response for testing if model failed to load (e.g. no GPU/PyTorch)
        if self.model is None:
            return self._generate_mock_response(message)

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
            
        if isinstance(message, str):
            message = [message]
            
        # Prepare all messages for batch processing
        texts = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            
            # Apply chat template
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback if template fails
                text = f"{system_prompt}\nUser: {msg}\nAssistant:"
                
            texts.append(text)

        # Tokenize with padding
        try:
            model_inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=4096
            ).to(self.model.device)
        except Exception as e:
            print(f"Tokenization error: {e}")
            return ["Error generating response"] * len(message), 0, 0.0

        # Generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        do_sample = kwargs.get("do_sample", True)
        
        tgps_show_var = kwargs.get("tgps_show", False)
        
        # Timer start
        start_time = time.time()
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            
        # Timer end
        generation_time = time.time() - start_time

        # Decode batch
        batch_outs = []
        token_len = 0
        
        for i, (input_ids, generated_sequence) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # Extract new tokens only
            input_len = len(input_ids)
            output_ids = generated_sequence[input_len:]
            
            token_len += len(output_ids)
            
            # Decode
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            batch_outs.append(content)

        # Return format handling
        result = batch_outs[0] if len(batch_outs) == 1 else batch_outs
        
        return result, token_len, generation_time

    def _generate_mock_response(self, message: str | List[str]) -> Tuple[Union[str, List[str]], int, float]:
        """Generate mock responses for testing without a GPU/Model"""
        import json
        import random
        
        if isinstance(message, str):
            messages = [message]
        else:
            messages = message
            
        responses = []
        for msg in messages:
            # Detect if this is a question generation prompt
            if "Generate" in msg and "question" in msg:
                 # Extract topic if possible
                topic = "General"
                if "topic:" in msg:
                    topic = msg.split("topic:")[-1].split()[0]
                
                # Create a valid dummy question
                dummy_q = {
                    "topic": topic,
                    "difficulty": "medium",
                    "question": f"This is a generated test question about {topic}?",
                    "choices": [
                        "A) Correct Answer",
                        "B) Wrong Answer 1", 
                        "C) Wrong Answer 2",
                        "D) Wrong Answer 3"
                    ],
                    "answer": "A",
                    "explanation": "This is a mock explanation for testing purposes."
                }
                responses.append(json.dumps(dummy_q, indent=2))
            else:
                responses.append("This is a mock response from Q-Agent (Model not loaded).")
        
        result = responses[0] if len(responses) == 1 else responses
        return result, 100, 0.5


if __name__ == "__main__":
    # Test Block
    try:
        model = QAgent()
        
        prompt = """Generate a hard MCQ based question on Number Series."""
        
        print("\nTesting Q-Agent generation...")
        response, tl, tm = model.generate_response(
            prompt,
            max_new_tokens=512,
            temperature=0.7
        )
        
        print(f"Response:\n{response}")
        print(f"Stats: {tl} tokens in {tm:.2f}s")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
