#!/usr/bin/python3

import re
import json
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

import sys
import os
import argparse
from pathlib import Path

# Robust import logic for question_model
# Strategy 1: Add parent directory to path (most likely scenario for agents/ subdirectory)
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

# Strategy 2: Add current directory to path
sys.path.insert(0, str(current_file.parent))

try:
    # Try importing assuming question_model is in the path now
    from question_model import QAgent
except ImportError:
    try:
        # Try relative import if we are in a package
        from .question_model import QAgent
    except ImportError:
        try:
            # Try importing as if it's in the 'agents' package
            from agents.question_model import QAgent
        except ImportError as e:
            print("CRITICAL ERROR: Could not import question_model.QAgent")
            print(f"Current path: {current_file}")
            print(f"Parent path: {parent_dir}")
            print(f"sys.path: {sys.path}")
            raise e


class QuestioningAgent:
    r"""Agent responsible for generating challenging MCQ questions"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = QAgent(**kwargs)
        
        # System prompts for question generation
        self.system_prompt1 = """You are an expert question generator specializing in creating challenging, valid multiple-choice questions.
Your questions should be:
1. Clear and unambiguous
2. Challenging but fair
3. Have exactly one correct answer
4. Include plausible distractors (wrong answers that seem reasonable)
5. Cover diverse aspects of the given topic"""

        self.system_prompt2 = """You are a master educator and assessment designer.
Create MCQ questions that:
- Test deep understanding, not just memorization
- Use realistic scenarios when applicable
- Have distractors based on common misconceptions
- Are appropriate for the specified difficulty level
- Follow best practices in question design"""

        self.system_prompt = self.system_prompt1 if select_prompt1 else self.system_prompt2

    def build_prompt(self, topic: str, difficulty: str = "medium", num_questions: int = 1) -> str:
        """Generate a prompt for creating MCQ questions"""
        
        difficulty_guidance = {
            "easy": "suitable for beginners, testing basic concepts and definitions",
            "medium": "requiring application of concepts and moderate reasoning",
            "hard": "demanding deep understanding, multi-step reasoning, or advanced knowledge",
            "expert": "requiring expert-level knowledge, complex analysis, or creative problem-solving"
        }
        
        guidance = difficulty_guidance.get(difficulty.lower(), difficulty_guidance["medium"])
        
        prompt = f"""Generate {num_questions} {'challenging' if difficulty in ['hard', 'expert'] else difficulty} multiple-choice question(s) on the topic: {topic}.

Difficulty Level: {difficulty.upper()} - {guidance}

Return your response as a valid JSON object (or array of objects if multiple questions) with this exact structure:

{{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "question": "Your question here ending with a question mark?",
    "choices": [
        "A) First option",
        "B) Second option", 
        "C) Third option",
        "D) Fourth option"
    ],
    "answer": "A",
    "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
}}

Requirements:
- Question must be clear, specific, and unambiguous
- All 4 choices must be plausible and grammatically consistent
- Exactly ONE choice must be correct
- Distractors should represent common misconceptions or errors
- Explanation should be concise but informative
- Return ONLY valid JSON, no additional text"""

        return prompt

    def generate_question(
        self, topic: str | List[str], difficulty: str = "medium", **kwargs
    ) -> str | List[str]:
        """Generate question(s) for the given topic(s)"""
        
        # Handle single topic or list of topics
        if isinstance(topic, str):
            topics = [topic]
        else:
            topics = topic
        
        # Build prompts for all topics
        prompts = [self.build_prompt(t, difficulty, num_questions=1) for t in topics]
        
        # Generate questions using the model
        responses, _, _ = self.agent.generate_response(
            prompts,
            system_prompt=self.system_prompt,
            **kwargs
        )
        
        # Return single response or list
        return responses if isinstance(topic, list) else responses

    def generate_batches(
        self, topics: List[str], difficulty: str = "medium", batch_size: int = 5, **kwargs
    ) -> List[str]:
        """Generate questions in batches for efficiency"""
        
        all_responses = []
        
        # If topics is a single string or empty, handle gracefully
        if isinstance(topics, str):
            topics = [topics]
        if not topics:
            return []

        # Ensure batch size is at least 1
        batch_size = max(1, batch_size)
        
        total_batches = (len(topics) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(topics), batch_size), desc="Generating questions"):
            batch = topics[i:i + batch_size]
            try:
                responses = self.generate_question(batch, difficulty=difficulty, **kwargs)
                
                # Handle both single response and list of responses
                if isinstance(responses, list):
                    all_responses.extend(responses)
                else:
                    all_responses.append(responses)
            except Exception as e:
                print(f"Error generating batch {i}: {e}")
                # Add placeholders to keep alignment if needed, or skip
                all_responses.extend([None] * len(batch))
        
        return all_responses

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if hasattr(self.agent, 'tokenizer') and self.agent.tokenizer:
            tokens = self.agent.tokenizer.encode(text)
            return len(tokens)
        return len(text.split())

    def filter_questions(self, questions: List[str | Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""Filter questions to ensure they are in the correct format"""
        
        def basic_checks(q: Dict[str, str]) -> bool:
            """Validate question structure"""
            required_keys = {"topic", "question", "choices", "answer"}
            
            # Check all required keys exist
            if not all(key in q for key in required_keys):
                return False
            
            # Check choices is a list of 4 items
            if not isinstance(q["choices"], list) or len(q["choices"]) != 4:
                return False
            
            # Check answer is one of A, B, C, D
            if q.get("answer") not in ["A", "B", "C", "D"]:
                return False
            
            # Check question ends with question mark
            if isinstance(q.get("question"), str) and not q["question"].strip().endswith("?"):
                return False
            
            return True
        
        filtered = []
        
        for q in questions:
            if q is None: continue
            
            try:
                # If it's a string, try to parse as JSON
                if isinstance(q, str):
                    # Extract JSON from markdown code blocks if present
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', q, re.DOTALL)
                    if json_match:
                        q = json_match.group(1)
                    
                    # Try to find JSON object in the string
                    json_match = re.search(r'\{.*\}', q, re.DOTALL)
                    if json_match:
                        q = json_match.group(0)
                    
                    q = json.loads(q)
                
                # Validate the question
                if basic_checks(q):
                    filtered.append(q)
                else:
                    # print(f"Warning: Question failed validation: {q.get('question', 'Unknown')[:50]}...")
                    pass
                    
            except json.JSONDecodeError as e:
                # print(f"Warning: Failed to parse JSON: {str(e)[:100]}")
                continue
            except Exception as e:
                # print(f"Warning: Unexpected error: {str(e)[:100]}")
                continue
        
        return filtered

    def save_questions(self, questions: List[Dict], file_path: str | Path) -> None:
        """Save generated questions to a JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(questions)} questions to {file_path}")

    def load_questions(self, file_path: str | Path) -> List[Dict]:
        """Load questions from a JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions


def main():
    parser = argparse.ArgumentParser(description="Run Q-Agent to generate questions")
    parser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Path to save generated questions")
    parser.add_argument("--num_questions", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--topics", type=str, nargs="+", default=None, help="List of topics")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "expert"], help="Difficulty level")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    # Default topics if none provided
    if not args.topics:
        default_topics = [
            "Logical Reasoning: Syllogisms",
            "Puzzles: Seating Arrangements (Linear, Circular)",
            "Blood Relations and Family Tree: Puzzles involving generations and family tree logic",
            "Alphanumeric Series: Mixed series questions"
        ]
        # Generate topics list of required length by cycling
        topics_to_use = []
        for i in range(args.num_questions):
            topics_to_use.append(default_topics[i % len(default_topics)])
    else:
        # Use provided topics, cycling if needed to match num_questions
        topics_to_use = []
        for i in range(args.num_questions):
            topics_to_use.append(args.topics[i % len(args.topics)])
    
    if args.verbose:
        print(f"Initializing Q-Agent with model: {args.model_name}")
        print(f"Generating {args.num_questions} questions on {len(set(topics_to_use))} unique topics")
        print(f"Difficulty: {args.difficulty}")
        print(f"Output file: {args.output_file}")

    # Initialize agent
    try:
        q_agent = QuestioningAgent(model_name=args.model_name)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    # Generate questions with timing
    import time
    start_time = time.time()
    
    raw_responses = q_agent.generate_batches(
        topics=topics_to_use,
        difficulty=args.difficulty,
        batch_size=5,
        max_new_tokens=512,
        temperature=0.7
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / max(1, len(raw_responses))
    
    # Filter and validate
    valid_questions = q_agent.filter_questions(raw_responses)
    
    if args.verbose:
        print(f"\nTime taken: {total_time:.2f}s (Average: {avg_time:.2f}s/question)")
        print(f"Generated {len(raw_responses)} raw responses")
        print(f"Valid questions: {len(valid_questions)} ({len(valid_questions)/len(raw_responses)*100:.1f}%)")
    
    # Save to file
    if valid_questions:
        q_agent.save_questions(valid_questions, args.output_file)
        if args.verbose:
            print(f"Sample Question:\n{json.dumps(valid_questions[0], indent=2)}")
    else:
        print("No valid questions generated.")


if __name__ == "__main__":
    main()
