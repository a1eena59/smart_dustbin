"""Pure burst-voting logic. Standard library only: no cv2, no model, no I/O."""

from dataclasses import dataclass
from typing import Literal

from wastevision.settings import CATEGORIES, CATEGORY_INDEX, Settings

FrameVerdict = tuple[str | None, float]


@dataclass(frozen=True)
class VoteOutcome:
    """The final decision for one burst."""

    category: str
    index: int
    confidence: float
    decided_by: Literal["vote", "fallback"]
    votes: dict[str, int]
    frames_used: int


def tally(frame_results: list[FrameVerdict], settings: Settings) -> VoteOutcome:
    """Combine per-frame verdicts into one decision, falling back when the evidence is weak."""
    votes = {c: 0 for c in CATEGORIES}
    confidences: dict[str, list[float]] = {c: [] for c in CATEGORIES}

    for category, confidence in frame_results:
        # Written as `not >=` so a NaN confidence abstains instead of slipping through.
        if category is None or not confidence >= settings.conf_floor:
            continue
        # An unknown label can only come from a mis-configured model; abstaining keeps
        # the bin working instead of crashing mid-sort.
        if category not in votes:
            continue
        votes[category] += 1
        confidences[category].append(confidence)

    frames_used = sum(votes.values())
    # sorted() is stable, so equal counts keep CATEGORIES order and the result is deterministic.
    ranked = sorted(CATEGORIES, key=lambda c: votes[c], reverse=True)
    winner, runner_up = ranked[0], ranked[1]

    if frames_used >= settings.min_valid_votes and votes[winner] - votes[runner_up] >= settings.min_margin:
        winning = confidences[winner]
        return VoteOutcome(
            category=winner,
            index=CATEGORY_INDEX[winner],
            confidence=sum(winning) / len(winning),
            decided_by="vote",
            votes=votes,
            frames_used=frames_used,
        )

    fallback = settings.fallback_category
    return VoteOutcome(
        category=fallback,
        index=CATEGORY_INDEX[fallback],
        confidence=0.0,
        decided_by="fallback",
        votes=votes,
        frames_used=frames_used,
    )
