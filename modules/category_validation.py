"""
Category Validation Gate (Rule-Based, Weighted Scoring) -- dipindahkan
langsung dari notebook riset, bukan dummy lagi.
"""

import re
from typing import Tuple

from modules._category_keywords import (
    HEALTH_PRIMARY,
    HEALTH_SECONDARY,
    HEALTH_PHRASE,
    POLITICAL_PRIMARY,
    POLITICAL_SECONDARY,
    POLITICAL_PHRASE,
    ECONOMY_PRIMARY,
    ECONOMY_SECONDARY,
    ECONOMY_PHRASE,
    NEGATIVE,
)

ALLOWED_CATEGORIES = ["Kesehatan", "Politik dan Pemerintahan", "Ekonomi"]

REJECTION_MESSAGE = (
    "Kategori berita berada di luar ruang lingkup penelitian. "
    "Penelitian ini hanya mendukung: Kesehatan, Politik dan Pemerintahan, "
    "atau Ekonomi."
)


def calculate_category_score(text, primary, secondary, phrases, negative):
    score = 0
    matched = set()
    phrase_found = False

    for phrase, weight in phrases.items():
        if phrase in text:
            score += weight
            matched.add(phrase)
            phrase_found = True

    for keyword, weight in primary.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            score += weight
            matched.add(keyword)

    for keyword, weight in secondary.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            score += weight
            matched.add(keyword)

    for keyword in negative:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            score -= 5

    return {
        "score": max(score, 0),
        "matched": matched,
        "unique": len(matched),
        "phrase": phrase_found,
    }


def validate_category(text: str) -> dict:
    clean_text = text.lower()

    health_result = calculate_category_score(
        clean_text,
        HEALTH_PRIMARY,
        HEALTH_SECONDARY,
        HEALTH_PHRASE,
        NEGATIVE["Kesehatan"],
    )
    politic_result = calculate_category_score(
        clean_text,
        POLITICAL_PRIMARY,
        POLITICAL_SECONDARY,
        POLITICAL_PHRASE,
        NEGATIVE["Politik"],
    )
    economy_result = calculate_category_score(
        clean_text,
        ECONOMY_PRIMARY,
        ECONOMY_SECONDARY,
        ECONOMY_PHRASE,
        NEGATIVE["Ekonomi"],
    )

    category_results = {
        "Kesehatan": health_result,
        "Politik dan Pemerintahan": politic_result,
        "Ekonomi": economy_result,
    }
    scores = {cat: res["score"] for cat, res in category_results.items()}
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_category = ranking[0][0]
    best_score = ranking[0][1]
    second_score = ranking[1][1] if len(ranking) > 1 else 0
    total = sum(scores.values())
    confidence = 0 if total == 0 else round(best_score / total * 100, 2)

    if best_score < 5:
        return {
            "category": None,
            "confidence": confidence,
            "scores": scores,
            "is_valid": False,
        }
    if best_score - second_score < 2:
        return {
            "category": None,
            "confidence": confidence,
            "scores": scores,
            "is_valid": False,
        }
    if (
        category_results[best_category]["unique"] < 2
        and not category_results[best_category]["phrase"]
    ):
        return {
            "category": None,
            "confidence": confidence,
            "scores": scores,
            "is_valid": False,
        }

    return {
        "category": best_category,
        "confidence": confidence,
        "scores": scores,
        "is_valid": True,
    }


def validate_category_rule_based(text: str) -> Tuple[bool, str]:
    result = validate_category(text)
    if result["is_valid"]:
        return True, result["category"]
    return False, REJECTION_MESSAGE
