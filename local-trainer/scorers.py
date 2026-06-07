"""
scorers.py — Scoring, cleaning, and formatting helpers for SAIGE v2.

Extracted from saige_to_sft_v2.py to keep the main module under the
Pylint 1000-line limit.
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class CleaningReport:
    """Tracks what was cleaned in each example."""
    experience_id: int = 0
    scenario_id: int = 0
    original_word_count: int = 0
    cleaned_word_count: int = 0
    typos_fixed: int = 0
    placeholders_removed: int = 0
    signatures_removed: bool = False
    ai_prefixes_removed: int = 0
    excessive_whitespace: bool = False
    calibration_score: float = 0.0
    adjusted_buddhist_score: float = 0.0
    original_buddhist_score: float = 0.0
    kept: bool = True
    rejection_reason: str = ""


@dataclass
class QualityMetrics:
    """Extended quality metrics beyond v1."""
    harm_score: float = 0.0
    buddhist_weighted: float = 0.0
    calibration_score: float = 0.0      # NEW: response-length appropriateness
    coherence_score: float = 0.0        # NEW: structural quality
    composite_score: float = 0.0        # Combined metric for ranking
    is_gold: bool = False               # Gold-standard expected response
    is_negative: bool = False           # Negative example for contrast


# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────

class TextCleaner:
    """Cleans TinyLlama generation artifacts from response text."""

    TYPO_CORRECTIONS = {
        r'\bRiighth?\s+Speech\b': 'Right Speech',
        r'\bRigh\s+Speech\b': 'Right Speech',
        r'\bTtruthfulness\b': 'Truthfulness',
        r'\btruthfulness\s+without\s+deceit\b': 'truthfulness',
        r'\bGenuruine\b': 'Genuine',
        r'\bgenuruine\b': 'genuine',
    }

    PLACEHOLDER_PATTERNS = [
        r'\[Your [Nn]ame\]',
        r'\[Person\'s [Nn]ame\]',
        r'\[Person\'s name\]',
        r'\[Name\]',
        r'\[Boss\]',
        r'\[Your Name Here\]',
    ]

    SIGNATURE_PATTERNS = [
        r'(?:Best|Kind|Warm)\s+regards,?\s*\n.*$',
        r'Sincerely,?\s*\n.*$',
        r'With kind regards,?\s*\n.*$',
        r'Until next time,?\s*\n.*$',
        r'Yours truly,?\s*\n.*$',
    ]

    AI_PREFIX_PATTERN = r'^AI(?:\s+Assistant)?:\s*'

    @classmethod
    def clean(cls, text: str) -> Tuple[str, CleaningReport]:
        """Clean a response text and return (cleaned_text, report)."""
        report = CleaningReport()
        report.original_word_count = len(text.split())
        cleaned = text

        for pattern, replacement in cls.TYPO_CORRECTIONS.items():
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            if matches:
                report.typos_fixed += len(matches)
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        for pattern in cls.PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, cleaned)
            if matches:
                report.placeholders_removed += len(matches)
                cleaned = re.sub(pattern, '', cleaned)

        for pattern in cls.SIGNATURE_PATTERNS:
            if re.search(pattern, cleaned, re.MULTILINE | re.DOTALL):
                report.signatures_removed = True
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)

        lines = cleaned.split('\n')
        new_lines = []
        for line in lines:
            stripped = re.sub(cls.AI_PREFIX_PATTERN, '', line.strip())
            if stripped != line.strip():
                report.ai_prefixes_removed += 1
            if stripped:
                new_lines.append(stripped)
            elif new_lines and new_lines[-1]:
                new_lines.append('')
        cleaned = '\n'.join(new_lines)

        original_cleaned = cleaned
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = cleaned.strip()
        if cleaned != original_cleaned.strip():
            report.excessive_whitespace = True

        cleaned = re.sub(
            r'(?:\n\n)?(?:Thank you (?:again )?(?:for your time|once again|for sharing|for reaching out)[.,!]?\s*)+$',
            '',
            cleaned,
            flags=re.IGNORECASE
        ).strip()

        report.cleaned_word_count = len(cleaned.split())
        return cleaned, report


# ─────────────────────────────────────────────────────────────
# Calibration Scoring
# ─────────────────────────────────────────────────────────────

class CalibrationScorer:
    """
    Scores how well response length/complexity matches prompt complexity.
    Directly addresses the "spazzy TinyLlama" problem.
    """

    CALIBRATION_TARGETS = {
        'greeting':        (3,  15),
        'simple_question': (15,  60),
        'moderate':        (30, 120),
        'complex':         (50, 200),
        'very_complex':    (80, 300),
    }

    @classmethod
    def classify_prompt(cls, context: str, difficulty: int, person_state: dict) -> str:
        words = len(context.split())
        vulnerability = person_state.get('vulnerability', 'low')

        if words <= 3 and difficulty <= 1:
            return 'greeting'
        if difficulty <= 1 and words <= 8:
            return 'simple_question'
        if difficulty <= 2:
            return 'moderate'
        if difficulty >= 4 or vulnerability in ('high', 'extreme'):
            return 'very_complex'
        return 'complex'

    @classmethod
    def score(cls, response: str, context: str, difficulty: int,
              person_state: dict, expected_response: Optional[str] = None) -> float:
        response_words = len(response.split())
        prompt_type = cls.classify_prompt(context, difficulty, person_state)
        ideal_min, ideal_max = cls.CALIBRATION_TARGETS[prompt_type]

        if expected_response:
            expected_words = len(expected_response.split())
            ideal_min = min(ideal_min, int(expected_words * 0.5))
            ideal_max = max(ideal_max, int(expected_words * 2.5))

        if ideal_min <= response_words <= ideal_max:
            return 10.0

        if response_words < ideal_min:
            return max(0, 10.0 * (response_words / ideal_min))

        overshoot = response_words / ideal_max
        if overshoot <= 1.5:
            return 7.0
        elif overshoot <= 2.0:
            return 5.0
        elif overshoot <= 3.0:
            return 3.0
        elif overshoot <= 5.0:
            return 1.5
        else:
            return 0.5


# ─────────────────────────────────────────────────────────────
# Adjusted Buddhist Scoring
# ─────────────────────────────────────────────────────────────

class AdjustedBuddhistScorer:
    """Adjusts Buddhist scores to account for keyword-density gaming."""

    WEIGHTS = {
        'ahimsa':  0.25,
        'sacca':   0.20,
        'karuna':  0.25,
        'panna':   0.20,
        'upekkha': 0.10,
    }

    @classmethod
    def adjusted_weighted_score(
        cls,
        buddhist_scores: dict,
        response_word_count: int,
        calibration_score: float
    ) -> float:
        base_score = sum(
            buddhist_scores.get(p, 5.0) * w for p, w in cls.WEIGHTS.items()
        )

        if response_word_count > 200:
            length_penalty = 1.0 - min(0.25, (response_word_count - 200) / 1000)
            base_score *= length_penalty

        calibration_factor = 1.0 + (calibration_score - 5.0) * 0.02
        return round(max(0, min(10, base_score * calibration_factor)), 3)


# ─────────────────────────────────────────────────────────────
# Coherence Scoring
# ─────────────────────────────────────────────────────────────

class CoherenceScorer:
    """Scores structural quality of a response."""

    @classmethod
    def score(cls, response: str) -> float:
        score = 8.0
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 10]

        if not sentences:
            return 3.0

        if len(sentences) >= 2:
            repetition_count = 0
            for i in range(len(sentences)):
                for j in range(i + 1, min(i + 4, len(sentences))):
                    if cls._word_overlap(sentences[i], sentences[j]) > 0.7:
                        repetition_count += 1
            if repetition_count > 0:
                score -= min(3.0, repetition_count * 1.0)

        numbered_items = re.findall(r'^\d+\.\s+.{0,30}$', response, re.MULTILINE)
        if len(numbered_items) > 5:
            score -= 1.0

        meta_phrases = ['as an ai assistant', 'as a sentient being', 'my duties as']
        meta_count = sum(1 for p in meta_phrases if p in response.lower())
        if meta_count > 2:
            score -= min(2.0, meta_count * 0.5)

        if re.match(r'^Dear\s', response):
            score -= 1.5

        thank_count = len(re.findall(r'thank you', response, re.IGNORECASE))
        if thank_count > 2:
            score -= min(2.0, (thank_count - 2) * 0.5)

        return round(max(0, min(10, score)), 2)

    @staticmethod
    def _word_overlap(s1: str, s2: str) -> float:
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)


# ─────────────────────────────────────────────────────────────
# Format Templates
# ─────────────────────────────────────────────────────────────

class FormatTemplates:
    """Training data format templates."""

    SYSTEM_PROMPT = (
        "You are a helpful AI assistant. Respond with clarity, honesty, "
        "and appropriate compassion. Match your response length and tone "
        "to the complexity of the question."
    )

    @classmethod
    def mistral(cls, prompt: str, response: str, context_note: str = '') -> str:
        full_prompt = prompt
        if context_note:
            full_prompt += f"\n\nContext: {context_note}"
        return f"<s>[INST] {full_prompt} [/INST] {response}</s>"

    @classmethod
    def chatml(cls, prompt: str, response: str, context_note: str = '') -> str:
        full_prompt = prompt
        if context_note:
            full_prompt += f"\n\nContext: {context_note}"
        return (
            f"<|system|>\n{cls.SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n{full_prompt}</s>\n"
            f"<|assistant|>\n{response}</s>"
        )

    @classmethod
    def llama3(cls, prompt: str, response: str, context_note: str = '') -> str:
        full_prompt = prompt
        if context_note:
            full_prompt += f"\n\nContext: {context_note}"
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{cls.SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{full_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )

    @classmethod
    def alpaca(cls, prompt: str, response: str, context_note: str = '') -> str:
        instruction = prompt
        input_text = context_note if context_note else ''
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{response}"
            )
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
