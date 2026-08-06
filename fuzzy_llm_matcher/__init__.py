"""fuzzy_llm_matcher: reliable fuzzy matching with optional LLM review.

Combines deterministic string similarity, score-margin based confidence
estimation, and optional LLM review for ambiguous cases -- the goal is
not just fuzzy matching, but a reliability layer that flags uncertain or
falsely confident matches.
"""

from .api import fuzzy_dissolve, fuzzy_join, fuzzy_join_geodataframes, match_geodataframes, match_tables
from .candidate_generation import generate_candidates
from .evaluation import EvaluationResult, evaluate_matches
from .fuzzy_scores import (
    add_geo_distance_score,
    compute_similarity_features,
    geo_distance_score,
    haversine_km,
)
from .geo_proximity import (
    TileBasemap,
    add_basemap,
    combined_score,
    sjoin_nearest_candidates,
)
from .llm_review import MockLLMClient, build_prompt, review_uncertain_pairs_with_llm
from .reliability import assign_reliability, false_confident_matches
from .simulation import simulate_dirty_entities

__version__ = "0.1.0"

__all__ = [
    "match_tables",
    "match_geodataframes",
    "fuzzy_join",
    "fuzzy_join_geodataframes",
    "fuzzy_dissolve",
    "sjoin_nearest_candidates",
    "combined_score",
    "add_basemap",
    "TileBasemap",
    "generate_candidates",
    "compute_similarity_features",
    "add_geo_distance_score",
    "geo_distance_score",
    "haversine_km",
    "assign_reliability",
    "false_confident_matches",
    "review_uncertain_pairs_with_llm",
    "MockLLMClient",
    "build_prompt",
    "evaluate_matches",
    "EvaluationResult",
    "simulate_dirty_entities",
]
