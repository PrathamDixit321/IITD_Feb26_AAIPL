#!/usr/bin/python3
"""
Test script to verify all components are working correctly
Run this to ensure your setup is ready for the competition
"""

import sys
import json
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*70)
    print("TEST 1: Checking Imports")
    print("="*70)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
        return False
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError as e:
        print(f"❌ PEFT import failed: {e}")
        return False
    
    try:
        import trl
        print(f"✅ TRL: {trl.__version__}")
    except ImportError as e:
        print(f"❌ TRL import failed: {e}")
        return False
    
    print("\n✅ All core dependencies imported successfully!")
    return True


def test_question_validation():
    """Test question validation logic"""
    print("\n" + "="*70)
    print("TEST 2: Question Validation")
    print("="*70)
    
    from evaluator import QuestionValidator
    
    # Valid question
    valid_question = {
        "topic": "Mathematics",
        "question": "What is 2 + 2?",
        "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
        "answer": "B",
        "explanation": "2 + 2 = 4"
    }
    
    result = QuestionValidator.validate_question(valid_question, 0)
    if result.is_valid:
        print("✅ Valid question accepted")
    else:
        print(f"❌ Valid question rejected: {result.errors}")
        return False
    
    # Invalid question (missing question mark)
    invalid_question = {
        "topic": "Mathematics",
        "question": "What is 2 + 2",  # Missing ?
        "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
        "answer": "B"
    }
    
    result = QuestionValidator.validate_question(invalid_question, 1)
    if not result.is_valid:
        print("✅ Invalid question rejected correctly")
    else:
        print("❌ Invalid question was accepted")
        return False
    
    # Invalid question (wrong number of choices)
    invalid_question2 = {
        "topic": "Mathematics",
        "question": "What is 2 + 2?",
        "choices": ["A) 3", "B) 4"],  # Only 2 choices
        "answer": "B"
    }
    
    result = QuestionValidator.validate_question(invalid_question2, 2)
    if not result.is_valid:
        print("✅ Question with wrong number of choices rejected")
    else:
        print("❌ Question with wrong number of choices was accepted")
        return False
    
    print("\n✅ Question validation working correctly!")
    return True


def test_answer_validation():
    """Test answer validation logic"""
    print("\n" + "="*70)
    print("TEST 3: Answer Validation")
    print("="*70)
    
    from evaluator import AnswerValidator
    
    # Valid correct answer
    valid_answer = {
        "answer": "B",
        "confidence": 0.95,
        "reasoning": "2 + 2 = 4"
    }
    
    result = AnswerValidator.validate_answer(valid_answer, "B", 0)
    if result.is_valid and result.is_correct:
        print("✅ Valid correct answer accepted")
    else:
        print(f"❌ Valid correct answer rejected: {result.errors}")
        return False
    
    # Valid incorrect answer
    valid_wrong_answer = {
        "answer": "A",
        "confidence": 0.5,
        "reasoning": "Guessing"
    }
    
    result = AnswerValidator.validate_answer(valid_wrong_answer, "B", 1)
    if result.is_valid and not result.is_correct:
        print("✅ Valid incorrect answer handled correctly")
    else:
        print("❌ Valid incorrect answer not handled correctly")
        return False
    
    # Invalid answer
    invalid_answer = {
        "answer": "Z",  # Invalid choice
        "confidence": 0.5
    }
    
    result = AnswerValidator.validate_answer(invalid_answer, "B", 2)
    if not result.is_valid:
        print("✅ Invalid answer rejected correctly")
    else:
        print("❌ Invalid answer was accepted")
        return False
    
    print("\n✅ Answer validation working correctly!")
    return True


def test_scoring():
    """Test scoring calculation"""
    print("\n" + "="*70)
    print("TEST 4: Scoring Calculation")
    print("="*70)
    
    from evaluator import MatchEvaluator
    
    # Create sample match data
    team_a_questions = [
        {
            "topic": "Math",
            "question": "What is 2 + 2?",
            "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
            "answer": "B",
            "explanation": "2 + 2 = 4"
        },
        {
            "topic": "Math",
            "question": "What is 3 + 3?",
            "choices": ["A) 5", "B) 6", "C) 7", "D) 8"],
            "answer": "B",
            "explanation": "3 + 3 = 6"
        }
    ]
    
    team_b_questions = [
        {
            "topic": "Math",
            "question": "What is 5 + 5?",
            "choices": ["A) 9", "B) 10", "C) 11", "D) 12"],
            "answer": "B",
            "explanation": "5 + 5 = 10"
        }
    ]
    
    # Team A answers (1 correct out of 1)
    team_a_answers = [
        {"answer": "B", "confidence": 0.95}
    ]
    
    # Team B answers (1 correct, 1 wrong out of 2)
    team_b_answers = [
        {"answer": "B", "confidence": 0.95},  # Correct
        {"answer": "A", "confidence": 0.50}   # Wrong
    ]
    
    evaluator = MatchEvaluator(min_valid_pct=50.0)
    result = evaluator.evaluate_match(
        team_a_name="Test Team A",
        team_b_name="Test Team B",
        team_a_questions=team_a_questions,
        team_b_questions=team_b_questions,
        team_a_answers=team_a_answers,
        team_b_answers=team_b_answers
    )
    
    # Verify calculations
    # Team A: 2 valid questions, Team B got 1 correct
    # Team A Q-score = (1 incorrect / 2) * 100 = 50.0
    # Team B A-score = (1 correct / 2) * 100 = 50.0
    
    print(f"Team A Q-Score: {result.team_a_q_score:.2f} (expected: 50.00)")
    print(f"Team B A-Score: {result.team_b_a_score:.2f} (expected: 50.00)")
    
    # Team B: 1 valid question, Team A got 1 correct
    # Team B Q-score = (0 incorrect / 1) * 100 = 0.0
    # Team A A-score = (1 correct / 1) * 100 = 100.0
    
    print(f"Team B Q-Score: {result.team_b_q_score:.2f} (expected: 0.00)")
    print(f"Team A A-Score: {result.team_a_a_score:.2f} (expected: 100.00)")
    
    if abs(result.team_a_q_score - 50.0) < 0.01 and \
       abs(result.team_b_a_score - 50.0) < 0.01 and \
       abs(result.team_b_q_score - 0.0) < 0.01 and \
       abs(result.team_a_a_score - 100.0) < 0.01:
        print("\n✅ Scoring calculation correct!")
        return True
    else:
        print("\n❌ Scoring calculation incorrect!")
        return False


def test_disqualification():
    """Test disqualification logic"""
    print("\n" + "="*70)
    print("TEST 5: Disqualification Logic")
    print("="*70)
    
    from evaluator import MatchEvaluator
    
    # Team A: 1 valid out of 3 questions (33.3% < 50%)
    team_a_questions = [
        {"topic": "Math", "question": "Valid?", "choices": ["A) 1", "B) 2", "C) 3", "D) 4"], "answer": "A"},
        {"topic": "Math", "question": "Invalid", "choices": ["A) 1", "B) 2"], "answer": "A"},  # Only 2 choices
        {"topic": "Math", "question": "Invalid", "choices": ["A) 1", "B) 2", "C) 3", "D) 4"], "answer": "Z"}  # Invalid answer
    ]
    
    # Team B: 2 valid out of 2 questions (100%)
    team_b_questions = [
        {"topic": "Math", "question": "Valid?", "choices": ["A) 1", "B) 2", "C) 3", "D) 4"], "answer": "A"},
        {"topic": "Math", "question": "Valid?", "choices": ["A) 1", "B) 2", "C) 3", "D) 4"], "answer": "B"}
    ]
    
    team_a_answers = [{"answer": "A"}, {"answer": "B"}]
    team_b_answers = [{"answer": "A"}]
    
    evaluator = MatchEvaluator(min_valid_pct=50.0)
    result = evaluator.evaluate_match(
        team_a_name="Team A (Should be DQ)",
        team_b_name="Team B (Should win)",
        team_a_questions=team_a_questions,
        team_b_questions=team_b_questions,
        team_a_answers=team_a_answers,
        team_b_answers=team_b_answers
    )
    
    if result.team_a_disqualified and not result.team_b_disqualified:
        print(f"✅ Team A correctly disqualified ({result.team_a_questions_valid_pct:.1f}% valid)")
        print(f"✅ Team B not disqualified ({result.team_b_questions_valid_pct:.1f}% valid)")
        print(f"✅ Winner: {result.winner}")
        return True
    else:
        print(f"❌ Disqualification logic failed")
        print(f"   Team A DQ: {result.team_a_disqualified} (should be True)")
        print(f"   Team B DQ: {result.team_b_disqualified} (should be False)")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("AAIPL SYSTEM TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Question Validation", test_question_validation),
        ("Answer Validation", test_answer_validation),
        ("Scoring Calculation", test_scoring),
        ("Disqualification Logic", test_disqualification)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for competition!")
    else:
        print("⚠️  SOME TESTS FAILED! Please fix issues before competing.")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
