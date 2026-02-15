#!/usr/bin/python3
"""
Match Orchestration System for AAIPL
Runs complete matches between teams with Q-agent and A-agent
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from question_agent import QuestioningAgent
from answer_agent import AnsweringAgent
from evaluator import MatchEvaluator


@dataclass
class TeamConfig:
    """Configuration for a team"""
    name: str
    q_agent_model_path: Optional[str] = None  # Path to fine-tuned Q-agent
    a_agent_model_path: Optional[str] = None  # Path to fine-tuned A-agent
    use_default_models: bool = True  # Use default models if paths not provided


class MatchOrchestrator:
    """Orchestrates matches between teams"""
    
    def __init__(
        self,
        num_questions: int = 10,
        difficulty: str = "medium",
        topics: Optional[List[str]] = None,
        min_valid_pct: float = 50.0,
        save_dir: str = "./match_results"
    ):
        """
        Args:
            num_questions: Number of questions each Q-agent should generate
            difficulty: Difficulty level (easy/medium/hard/expert)
            topics: List of topics to generate questions on (random if None)
            min_valid_pct: Minimum percentage of valid questions
            save_dir: Directory to save match results
        """
        self.num_questions = num_questions
        self.difficulty = difficulty
        self.topics = topics or [
            "Mathematics", "Probability", "Algorithms", "Data Structures",
            "Machine Learning", "Physics", "Chemistry", "Biology",
            "History", "Geography", "Logic", "Number Series"
        ]
        self.evaluator = MatchEvaluator(min_valid_pct=min_valid_pct)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def load_team_agents(self, team_config: TeamConfig):
        """Load Q-agent and A-agent for a team"""
        
        print(f"\nLoading agents for {team_config.name}...")
        
        # Load Q-agent
        if team_config.use_default_models or not team_config.q_agent_model_path:
            print(f"  Loading default Q-agent...")
            q_agent = QuestioningAgent()
        else:
            print(f"  Loading fine-tuned Q-agent from {team_config.q_agent_model_path}...")
            # TODO: Implement loading fine-tuned models
            q_agent = QuestioningAgent()
        
        # Load A-agent
        if team_config.use_default_models or not team_config.a_agent_model_path:
            print(f"  Loading default A-agent...")
            a_agent = AnsweringAgent()
        else:
            print(f"  Loading fine-tuned A-agent from {team_config.a_agent_model_path}...")
            # TODO: Implement loading fine-tuned models
            a_agent = AnsweringAgent()
        
        return q_agent, a_agent
    
    def generate_questions(
        self,
        q_agent: QuestioningAgent,
        team_name: str,
        num_questions: int,
        topics: List[str],
        difficulty: str
    ) -> List[str]:
        """Generate questions using Q-agent"""
        
        print(f"\n{team_name} Q-Agent generating {num_questions} questions...")
        
        # Select topics (cycle through if needed)
        selected_topics = []
        for i in range(num_questions):
            selected_topics.append(topics[i % len(topics)])
        
        # Generate questions
        start_time = time.time()
        questions = q_agent.generate_batches(
            topics=selected_topics,
            difficulty=difficulty,
            batch_size=5,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        generation_time = time.time() - start_time
        
        print(f"  Generated {len(questions)} questions in {generation_time:.2f}s")
        
        return questions
    
    def generate_answers(
        self,
        a_agent: AnsweringAgent,
        team_name: str,
        questions: List[Dict]
    ) -> List[str]:
        """Generate answers using A-agent"""
        
        print(f"\n{team_name} A-Agent answering {len(questions)} questions...")
        
        # Generate answers
        start_time = time.time()
        answers = a_agent.answer_batches(
            questions=questions,
            batch_size=5,
            max_new_tokens=256,
            temperature=0.3,  # Lower temperature for more consistent answers
            top_p=0.9,
            do_sample=True
        )
        generation_time = time.time() - start_time
        
        print(f"  Generated {len(answers)} answers in {generation_time:.2f}s")
        
        return answers
    
    def run_match(
        self,
        team_a_config: TeamConfig,
        team_b_config: TeamConfig,
        match_id: Optional[str] = None
    ):
        """
        Run a complete match between two teams
        
        Match flow:
        1. Load both teams' agents
        2. Team A's Q-agent generates questions
        3. Team B's Q-agent generates questions
        4. Team A's A-agent answers Team B's questions
        5. Team B's A-agent answers Team A's questions
        6. Evaluate and score the match
        """
        
        if match_id is None:
            match_id = f"{team_a_config.name}_vs_{team_b_config.name}_{int(time.time())}"
        
        print("\n" + "="*80)
        print(f"STARTING MATCH: {team_a_config.name} vs {team_b_config.name}")
        print(f"Match ID: {match_id}")
        print(f"Questions per team: {self.num_questions}")
        print(f"Difficulty: {self.difficulty}")
        print("="*80)
        
        # Load agents
        team_a_q_agent, team_a_a_agent = self.load_team_agents(team_a_config)
        team_b_q_agent, team_b_a_agent = self.load_team_agents(team_b_config)
        
        # Phase 1: Generate questions
        print("\n" + "-"*80)
        print("PHASE 1: QUESTION GENERATION")
        print("-"*80)
        
        team_a_questions_raw = self.generate_questions(
            team_a_q_agent,
            team_a_config.name,
            self.num_questions,
            self.topics,
            self.difficulty
        )
        
        team_b_questions_raw = self.generate_questions(
            team_b_q_agent,
            team_b_config.name,
            self.num_questions,
            self.topics,
            self.difficulty
        )
        
        # Filter valid questions for answering phase
        team_a_questions_filtered = team_a_q_agent.filter_questions(team_a_questions_raw)
        team_b_questions_filtered = team_b_q_agent.filter_questions(team_b_questions_raw)
        
        print(f"\n{team_a_config.name}: {len(team_a_questions_filtered)}/{len(team_a_questions_raw)} valid questions")
        print(f"{team_b_config.name}: {len(team_b_questions_filtered)}/{len(team_b_questions_raw)} valid questions")
        
        # Phase 2: Generate answers
        print("\n" + "-"*80)
        print("PHASE 2: ANSWER GENERATION")
        print("-"*80)
        
        team_a_answers_raw = self.generate_answers(
            team_a_a_agent,
            team_a_config.name,
            team_b_questions_filtered
        )
        
        team_b_answers_raw = self.generate_answers(
            team_b_a_agent,
            team_b_config.name,
            team_a_questions_filtered
        )
        
        # Phase 3: Evaluation
        print("\n" + "-"*80)
        print("PHASE 3: EVALUATION")
        print("-"*80)
        
        result = self.evaluator.evaluate_match(
            team_a_name=team_a_config.name,
            team_b_name=team_b_config.name,
            team_a_questions=team_a_questions_raw,
            team_b_questions=team_b_questions_raw,
            team_a_answers=team_a_answers_raw,
            team_b_answers=team_b_answers_raw
        )
        
        # Print summary
        self.evaluator.print_match_summary(result)
        
        # Save results
        result_path = self.save_dir / f"{match_id}.json"
        self.evaluator.save_match_result(result, str(result_path))
        
        # Save detailed data
        detailed_data = {
            "match_id": match_id,
            "team_a": {
                "name": team_a_config.name,
                "questions_raw": team_a_questions_raw,
                "questions_filtered": team_a_questions_filtered,
                "answers": team_a_answers_raw
            },
            "team_b": {
                "name": team_b_config.name,
                "questions_raw": team_b_questions_raw,
                "questions_filtered": team_b_questions_filtered,
                "answers": team_b_answers_raw
            }
        }
        
        detailed_path = self.save_dir / f"{match_id}_detailed.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed match data saved to: {detailed_path}")
        
        return result
    
    def run_tournament(self, teams: List[TeamConfig]):
        """
        Run a round-robin tournament between multiple teams
        
        Args:
            teams: List of team configurations
        """
        
        print("\n" + "="*80)
        print(f"STARTING TOURNAMENT: {len(teams)} teams")
        print("="*80)
        
        results = []
        
        # Round-robin: each team plays every other team
        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:
                result = self.run_match(team_a, team_b)
                results.append(result)
        
        # Calculate standings
        print("\n" + "="*80)
        print("TOURNAMENT STANDINGS")
        print("="*80)
        
        team_scores = {}
        for team in teams:
            team_scores[team.name] = {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_score": 0.0,
                "matches": 0
            }
        
        for result in results:
            team_a = result.team_a_name
            team_b = result.team_b_name
            
            if not result.team_a_disqualified and not result.team_b_disqualified:
                team_scores[team_a]["total_score"] += result.team_a_total_score
                team_scores[team_b]["total_score"] += result.team_b_total_score
                team_scores[team_a]["matches"] += 1
                team_scores[team_b]["matches"] += 1
                
                if result.winner == team_a:
                    team_scores[team_a]["wins"] += 1
                    team_scores[team_b]["losses"] += 1
                elif result.winner == team_b:
                    team_scores[team_b]["wins"] += 1
                    team_scores[team_a]["losses"] += 1
                else:
                    team_scores[team_a]["ties"] += 1
                    team_scores[team_b]["ties"] += 1
        
        # Sort by wins, then by average score
        standings = sorted(
            team_scores.items(),
            key=lambda x: (x[1]["wins"], x[1]["total_score"] / max(x[1]["matches"], 1)),
            reverse=True
        )
        
        print("\nRank | Team | W-L-T | Avg Score")
        print("-" * 50)
        for rank, (team_name, stats) in enumerate(standings, 1):
            avg_score = stats["total_score"] / max(stats["matches"], 1)
            print(f"{rank:4d} | {team_name:20s} | {stats['wins']}-{stats['losses']}-{stats['ties']} | {avg_score:8.2f}")
        
        # Save tournament results
        tournament_result = {
            "teams": [team.name for team in teams],
            "standings": standings,
            "matches": [result.__dict__ for result in results]
        }
        
        tournament_path = self.save_dir / f"tournament_{int(time.time())}.json"
        with open(tournament_path, 'w', encoding='utf-8') as f:
            json.dump(tournament_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nTournament results saved to: {tournament_path}")
        
        return standings


if __name__ == "__main__":
    # Example: Run a match between two teams
    
    team_a = TeamConfig(
        name="Team Alpha",
        use_default_models=True
    )
    
    team_b = TeamConfig(
        name="Team Beta",
        use_default_models=True
    )
    
    # Create orchestrator
    orchestrator = MatchOrchestrator(
        num_questions=5,  # Start small for testing
        difficulty="medium",
        min_valid_pct=50.0,
        save_dir="./match_results"
    )
    
    # Run a single match
    print("\n" + "="*80)
    print("EXAMPLE: Running a single match")
    print("="*80)
    
    result = orchestrator.run_match(team_a, team_b)
    
    # Example: Run a tournament
    # Uncomment to run tournament with multiple teams
    """
    teams = [
        TeamConfig(name="Team Alpha", use_default_models=True),
        TeamConfig(name="Team Beta", use_default_models=True),
        TeamConfig(name="Team Gamma", use_default_models=True),
    ]
    
    standings = orchestrator.run_tournament(teams)
    """
