#!/usr/bin/python3
"""
Quick Start Script for AAIPL
Demonstrates basic usage of all components
"""

def demo_question_generation():
    """Demo: Generate questions using Q-Agent"""
    print("\n" + "="*70)
    print("DEMO 1: Question Generation")
    print("="*70)
    
    from question_agent import QuestioningAgent
    
    # Initialize Q-Agent
    print("\nInitializing Q-Agent...")
    q_agent = QuestioningAgent()
    
    # Generate a single question
    print("\nGenerating a hard question on 'Machine Learning'...")
    question = q_agent.generate_question(
        topic="Machine Learning",
        difficulty="hard",
        max_new_tokens=512,
        temperature=0.7
    )
    
    print(f"\nGenerated Question:\n{question}\n")
    
    # Validate the question
    print("Validating question format...")
    filtered = q_agent.filter_questions([question])
    
    if filtered:
        print("✅ Question is valid!")
        import json
        print(json.dumps(filtered[0], indent=2))
    else:
        print("❌ Question failed validation")


def demo_answer_generation():
    """Demo: Answer questions using A-Agent"""
    print("\n" + "="*70)
    print("DEMO 2: Answer Generation")
    print("="*70)
    
    from answer_agent import AnsweringAgent
    
    # Initialize A-Agent
    print("\nInitializing A-Agent...")
    a_agent = AnsweringAgent()
    
    # Sample question
    question = {
        "topic": "Mathematics",
        "question": "What is the next number in the sequence: 2, 4, 8, 16, ?",
        "choices": [
            "A) 24",
            "B) 28",
            "C) 32",
            "D) 36"
        ],
        "answer": "C",
        "explanation": "Powers of 2: 2^1, 2^2, 2^3, 2^4, 2^5 = 32"
    }
    
    print("\nQuestion to answer:")
    print(f"  {question['question']}")
    for choice in question['choices']:
        print(f"    {choice}")
    
    # Generate answer
    print("\nGenerating answer...")
    answer = a_agent.answer_question(question, max_new_tokens=256)
    
    print(f"\nGenerated Answer:\n{answer}\n")
    
    # Validate the answer
    print("Validating answer format...")
    filtered = a_agent.filter_answers([answer])
    
    if filtered:
        print("✅ Answer is valid!")
        import json
        print(json.dumps(filtered[0], indent=2))
    else:
        print("❌ Answer failed validation")


def demo_evaluation():
    """Demo: Evaluate a match"""
    print("\n" + "="*70)
    print("DEMO 3: Match Evaluation")
    print("="*70)
    
    from evaluator import MatchEvaluator
    
    # Sample match data
    team_a_questions = [
        {
            "topic": "Math",
            "question": "What is 5 × 6?",
            "choices": ["A) 28", "B) 30", "C) 32", "D) 36"],
            "answer": "B",
            "explanation": "5 × 6 = 30"
        },
        {
            "topic": "Math",
            "question": "What is 7 + 8?",
            "choices": ["A) 14", "B) 15", "C) 16", "D) 17"],
            "answer": "B",
            "explanation": "7 + 8 = 15"
        }
    ]
    
    team_b_questions = [
        {
            "topic": "Science",
            "question": "What is the chemical symbol for water?",
            "choices": ["A) H2O", "B) CO2", "C) O2", "D) N2"],
            "answer": "A",
            "explanation": "Water is H2O"
        }
    ]
    
    # Team A answers Team B's question
    team_a_answers = [
        {"answer": "A", "confidence": 0.99, "reasoning": "Water is H2O"}
    ]
    
    # Team B answers Team A's questions
    team_b_answers = [
        {"answer": "B", "confidence": 0.95, "reasoning": "5 × 6 = 30"},
        {"answer": "A", "confidence": 0.60, "reasoning": "Guessing"}  # Wrong!
    ]
    
    # Evaluate
    print("\nEvaluating match...")
    evaluator = MatchEvaluator(min_valid_pct=50.0)
    
    result = evaluator.evaluate_match(
        team_a_name="Team Alpha",
        team_b_name="Team Beta",
        team_a_questions=team_a_questions,
        team_b_questions=team_b_questions,
        team_a_answers=team_a_answers,
        team_b_answers=team_b_answers
    )
    
    # Print results
    evaluator.print_match_summary(result)


def demo_full_match():
    """Demo: Run a complete match (requires models)"""
    print("\n" + "="*70)
    print("DEMO 4: Full Match Simulation")
    print("="*70)
    print("\nThis demo requires loading the full models.")
    print("It may take a few minutes and requires GPU access.")
    
    response = input("\nDo you want to run a full match? (y/n): ")
    
    if response.lower() != 'y':
        print("Skipping full match demo.")
        return
    
    from match_orchestrator import MatchOrchestrator, TeamConfig
    
    # Define teams
    team_a = TeamConfig(name="Team Alpha", use_default_models=True)
    team_b = TeamConfig(name="Team Beta", use_default_models=True)
    
    # Create orchestrator
    print("\nInitializing match orchestrator...")
    orchestrator = MatchOrchestrator(
        num_questions=3,  # Small number for demo
        difficulty="medium",
        min_valid_pct=50.0,
        save_dir="./demo_results"
    )
    
    # Run match
    print("\nRunning match...")
    result = orchestrator.run_match(team_a, team_b)
    
    print(f"\n✅ Match complete! Results saved to ./demo_results/")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("AMD AI PREMIER LEAGUE - QUICK START DEMOS")
    print("="*70)
    
    demos = [
        ("Question Generation", demo_question_generation),
        ("Answer Generation", demo_answer_generation),
        ("Match Evaluation", demo_evaluation),
        ("Full Match Simulation", demo_full_match)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos)+1}. Run all demos")
    print(f"  0. Exit")
    
    while True:
        try:
            choice = input(f"\nSelect demo (0-{len(demos)+1}): ").strip()
            
            if choice == "0":
                print("\nExiting. Good luck with the competition! 🚀")
                break
            
            choice_num = int(choice)
            
            if choice_num == len(demos) + 1:
                # Run all demos
                for name, demo_func in demos:
                    try:
                        demo_func()
                    except Exception as e:
                        print(f"\n❌ Error in {name}: {e}")
                        import traceback
                        traceback.print_exc()
                break
            
            elif 1 <= choice_num <= len(demos):
                # Run selected demo
                name, demo_func = demos[choice_num - 1]
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Error in {name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Invalid choice. Please enter 0-{len(demos)+1}")
        
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting. Good luck with the competition! 🚀")
            break
    
    print("\n" + "="*70)
    print("For more information, see:")
    print("  - README.md - Complete project guide")
    print("  - TRAINING_GUIDE.md - Training instructions")
    print("  - IMPLEMENTATION_SUMMARY.md - What we built")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
