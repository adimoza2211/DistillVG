from __future__ import annotations


def _pick_augmented_phrase(base_phrase: str, candidates: tuple[str, ...], strategy: str) -> str:
    if not candidates:
        return base_phrase

    normalized = tuple(candidate.strip() for candidate in candidates if candidate.strip())
    if not normalized:
        return base_phrase

    if strategy == "last":
        return normalized[-1]
    if strategy == "longest":
        return max(normalized, key=len)
    return normalized[0]


def build_verifier_queries(
    base_phrases: list[str],
    augmented_phrase_sets: list[tuple[str, ...]] | None,
    use_augmented: bool,
    selection_strategy: str,
) -> list[str]:
    queries: list[str] = []
    for index, base_phrase in enumerate(base_phrases):
        phrase = base_phrase.strip()
        if use_augmented and augmented_phrase_sets is not None and index < len(augmented_phrase_sets):
            phrase = _pick_augmented_phrase(
                base_phrase=phrase,
                candidates=augmented_phrase_sets[index],
                strategy=selection_strategy,
            )
        prompt = f"Does this image crop match the expression: '{phrase}'? Answer only True or False."
        queries.append(prompt)

    return queries


def build_stage2_verifier_queries(phrases: list[str], top_k: int) -> list[str]:
    queries: list[str] = []
    for phrase in phrases:
        prompt = (
            f"The red rectangle in this image highlights a specific region. "
            f"How well does the highlighted region match the expression '{phrase}'? "
            "Answer with exactly one word from: Excellent, Good, Partial, No."
        )
        queries.extend([prompt] * top_k)
    return queries
