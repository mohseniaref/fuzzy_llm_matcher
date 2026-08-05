"""fuzzy_llm_matcher: reliable fuzzy matching with optional LLM review.

Combines deterministic string similarity, score-margin based confidence
estimation, and optional LLM review for ambiguous cases -- the goal is
not just fuzzy matching, but a reliability layer that flags uncertain or
falsely confident matches.
"""

from .api import match_tables
from .candidate_generation import generate_candidates
from .evaluation import EvaluationResult, evaluate_matches
from .fuzzy_scores import compute_similarity_features
from .llm_review import MockLLMClient, review_uncertain_pairs_with_llm
from .reliability import assign_reliability, false_confident_matches
from .simulation import simulate_dirty_entities

__version__ = "0.1.0"

__all__ = [
    "match_tables",
    "generate_candidates",
    "compute_similarity_features",
    "assign_reliability",
    "false_confident_matches",
    "review_uncertain_pairs_with_llm",
    "MockLLMClient",
    "evaluate_matches",
    "EvaluationResult",
    "simulate_dirty_entities",
]
