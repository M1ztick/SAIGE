#!/usr/bin/env python3
"""
saige_to_sft_v2.py — SAIGE Training Data Cleaner & Improver (v2)

Improvements over v1:
  1. Text cleanup: strips typos, placeholders, letter signatures, AI: prefixes
  2. Calibration scoring: penalizes verbose responses to simple prompts
  3. Length-aware Buddhist scoring: adjusts for keyword-density gaming
  4. Gold-standard injection: includes expected_response as ideal examples
  5. Negative examples: includes clearly bad responses (labeled) for DPO/contrast
  6. Deduplication: picks the best experience per scenario instead of all of them
  7. Format flexibility: outputs Mistral, ChatML, Llama3, or Alpaca formats
  8. Detailed diagnostics: per-example quality report

Usage:
    python saige_to_sft_v2.py --db saige.db
    python saige_to_sft_v2.py --db saige.db --format chatml --include-negatives
    python saige_to_sft_v2.py --db saige.db --best-per-scenario --include-gold
    python saige_to_sft_v2.py --db saige.db --diagnostics-only
"""

import sqlite3
import json
import csv
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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

    # Known typos from TinyLlama
    TYPO_CORRECTIONS = {
        r'\bRiighth?\s+Speech\b': 'Right Speech',
        r'\bRigh\s+Speech\b': 'Right Speech',
        r'\bTtruthfulness\b': 'Truthfulness',
        r'\btruthfulness\s+without\s+deceit\b': 'truthfulness',
        r'\bGeneruine\b': 'Genuine',
        r'\bgeneruine\b': 'genuine',
    }

    # Placeholder patterns to remove
    PLACEHOLDER_PATTERNS = [
        r'\[Your [Nn]ame\]',
        r'\[Person\'s [Nn]ame\]',
        r'\[Person\'s name\]',
        r'\[Name\]',
        r'\[Boss\]',
        r'\[Your Name Here\]',
    ]

    # Letter-style sign-offs that shouldn't be in chat responses
    SIGNATURE_PATTERNS = [
        r'(?:Best|Kind|Warm)\s+regards,?\s*\n.*$',
        r'Sincerely,?\s*\n.*$',
        r'With kind regards,?\s*\n.*$',
        r'Until next time,?\s*\n.*$',
        r'Yours truly,?\s*\n.*$',
    ]

    # "AI:" prefix repetition
    AI_PREFIX_PATTERN = r'^AI(?:\s+Assistant)?:\s*'

    @classmethod
    def clean(cls, text: str) -> Tuple[str, CleaningReport]:
        """Clean a response text and return (cleaned_text, report)."""
        report = CleaningReport()
        report.original_word_count = len(text.split())
        cleaned = text

        # 1. Fix known typos
        for pattern, replacement in cls.TYPO_CORRECTIONS.items():
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            if matches:
                report.typos_fixed += len(matches)
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # 2. Remove placeholders
        for pattern in cls.PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, cleaned)
            if matches:
                report.placeholders_removed += len(matches)
                cleaned = re.sub(pattern, '', cleaned)

        # 3. Remove letter-style signatures
        for pattern in cls.SIGNATURE_PATTERNS:
            if re.search(pattern, cleaned, re.MULTILINE | re.DOTALL):
                report.signatures_removed = True
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)

        # 4. Remove repeated "AI:" prefixes (keep first one's content)
        lines = cleaned.split('\n')
        new_lines = []
        for line in lines:
            stripped = re.sub(cls.AI_PREFIX_PATTERN, '', line.strip())
            if stripped != line.strip():
                report.ai_prefixes_removed += 1
            if stripped:
                new_lines.append(stripped)
            elif new_lines and new_lines[-1]:  # preserve single blank lines
                new_lines.append('')
        cleaned = '\n'.join(new_lines)

        # 5. Collapse excessive whitespace / blank lines
        original_cleaned = cleaned
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = cleaned.strip()
        if cleaned != original_cleaned.strip():
            report.excessive_whitespace = True

        # 6. Remove trailing "Thank you" filler
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
    This directly addresses the "spazzy TinyLlama" problem.
    """

    # Target word counts based on scenario characteristics
    # (difficulty_level, prompt_word_count) -> (ideal_min, ideal_max)
    CALIBRATION_TARGETS = {
        # Simple greetings: 3-15 words
        'greeting': (3, 15),
        # Simple questions: 15-60 words
        'simple_question': (15, 60),
        # Moderate questions: 30-120 words
        'moderate': (30, 120),
        # Complex/emotional: 50-200 words
        'complex': (50, 200),
        # Very complex: 80-300 words
        'very_complex': (80, 300),
    }

    @classmethod
    def classify_prompt(cls, context: str, difficulty: int, person_state: dict) -> str:
        """Classify prompt type for calibration targeting."""
        words = len(context.split())
        vulnerability = person_state.get('vulnerability', 'low')
        mood = person_state.get('mood', 'neutral')

        # Simple greetings and acknowledgments
        if words <= 3 and difficulty <= 1:
            return 'greeting'

        # Simple factual questions
        if difficulty <= 1 and words <= 8:
            return 'simple_question'

        # Moderate complexity
        if difficulty <= 2:
            return 'moderate'

        # Complex or emotionally sensitive
        if difficulty >= 4 or vulnerability in ('high', 'extreme'):
            return 'very_complex'

        return 'complex'

    @classmethod
    def score(cls, response: str, context: str, difficulty: int,
              person_state: dict, expected_response: Optional[str] = None) -> float:
        """
        Score response calibration (0-10).
        10 = perfectly calibrated length
        0 = wildly miscalibrated
        """
        response_words = len(response.split())
        prompt_type = cls.classify_prompt(context, difficulty, person_state)
        ideal_min, ideal_max = cls.CALIBRATION_TARGETS[prompt_type]

        # If we have an expected response, use it as additional calibration anchor
        if expected_response:
            expected_words = len(expected_response.split())
            # Expand the ideal range slightly around the expected length
            ideal_min = min(ideal_min, int(expected_words * 0.5))
            ideal_max = max(ideal_max, int(expected_words * 2.5))

        # Score based on how close we are to the ideal range
        if ideal_min <= response_words <= ideal_max:
            return 10.0  # Perfect calibration

        if response_words < ideal_min:
            # Too short — moderate penalty
            ratio = response_words / ideal_min
            return max(0, 10.0 * ratio)

        # Too long — this is the main TinyLlama problem
        overshoot = response_words / ideal_max
        if overshoot <= 1.5:
            return 7.0  # Slightly over is OK
        elif overshoot <= 2.0:
            return 5.0  # Notably over
        elif overshoot <= 3.0:
            return 3.0  # Way too long
        elif overshoot <= 5.0:
            return 1.5  # Extremely verbose
        else:
            return 0.5  # Absurdly long


# ─────────────────────────────────────────────────────────────
# Adjusted Buddhist Scoring
# ─────────────────────────────────────────────────────────────

class AdjustedBuddhistScorer:
    """
    Adjusts Buddhist principle scores to account for keyword-density gaming.
    Longer responses naturally score higher on keyword matching — this corrects for that.
    """

    WEIGHTS = {
        'ahimsa': 0.25,
        'sacca': 0.20,
        'karuna': 0.25,
        'panna': 0.20,
        'upekkha': 0.10,
    }

    @classmethod
    def adjusted_weighted_score(
        cls,
        buddhist_scores: dict,
        response_word_count: int,
        calibration_score: float
    ) -> float:
        """
        Calculate a weighted Buddhist score adjusted for:
        1. Response length (diminishing returns for verbosity)
        2. Calibration quality (well-calibrated responses get a bonus)
        """
        # Base weighted score
        base_score = sum(
            buddhist_scores.get(principle, 5.0) * weight
            for principle, weight in cls.WEIGHTS.items()
        )

        # Length adjustment: responses over 200 words get diminishing returns
        # This prevents verbose responses from gaming the keyword scorer
        if response_word_count > 200:
            length_penalty = 1.0 - min(0.25, (response_word_count - 200) / 1000)
            base_score *= length_penalty

        # Calibration bonus: well-calibrated responses get a small boost
        calibration_factor = 1.0 + (calibration_score - 5.0) * 0.02  # ±10% range
        adjusted = base_score * calibration_factor

        return round(max(0, min(10, adjusted)), 3)


# ─────────────────────────────────────────────────────────────
# Coherence Scoring
# ─────────────────────────────────────────────────────────────

class CoherenceScorer:
    """
    Scores structural quality of a response.
    Catches: repetition, broken formatting, nonsensical structure.
    """

    @classmethod
    def score(cls, response: str) -> float:
        """Score coherence (0-10)."""
        score = 8.0  # Start optimistic

        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return 3.0

        # 1. Repetition detection — compare sentence pairs
        if len(sentences) >= 2:
            repetition_count = 0
            for i in range(len(sentences)):
                for j in range(i + 1, min(i + 4, len(sentences))):
                    sim = cls._word_overlap(sentences[i], sentences[j])
                    if sim > 0.7:
                        repetition_count += 1
            if repetition_count > 0:
                score -= min(3.0, repetition_count * 1.0)

        # 2. Numbered list without substance (TinyLlama loves empty lists)
        numbered_items = re.findall(r'^\d+\.\s+.{0,30}$', response, re.MULTILINE)
        if len(numbered_items) > 5:
            score -= 1.0

        # 3. Meta-commentary about being an AI (instead of actually helping)
        # Deliberately excludes substantive Buddhist references ('right speech',
        # 'buddhist ethics') that may appear in legitimate SAIGE responses.
        meta_phrases = [
            'as an ai assistant',
            'as a sentient being',
            'my duties as',
        ]
        meta_count = sum(1 for p in meta_phrases if p in response.lower())
        if meta_count > 2:
            score -= min(2.0, meta_count * 0.5)

        # 4. Starts with "Dear" or letter format for a chat response
        if re.match(r'^Dear\s', response):
            score -= 1.5

        # 5. Multiple "Thank you" closings
        thank_count = len(re.findall(r'thank you', response, re.IGNORECASE))
        if thank_count > 2:
            score -= min(2.0, (thank_count - 2) * 0.5)

        return round(max(0, min(10, score)), 2)

    @staticmethod
    def _word_overlap(s1: str, s2: str) -> float:
        """Jaccard similarity between two strings (intersection / union)."""
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


# ─────────────────────────────────────────────────────────────
# Main Converter
# ─────────────────────────────────────────────────────────────

class SAIGEv2Converter:
    """v2 converter with cleaning, calibration, and quality improvements."""

    FORMAT_MAP = {
        'mistral': FormatTemplates.mistral,
        'chatml': FormatTemplates.chatml,
        'llama3': FormatTemplates.llama3,
        'alpaca': FormatTemplates.alpaca,
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cleaning_reports: List[CleaningReport] = []

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        print(f"✅ Connected to: {self.db_path}")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def load_scenarios(self) -> Dict[int, dict]:
        """Load all scenarios keyed by ID."""
        scenarios = {}
        for row in self.conn.execute('SELECT * FROM scenarios'):
            scenarios[row['id']] = {
                'id': row['id'],
                'context': row['context'],
                'person_state': json.loads(row['person_state']) if row['person_state'] else {},
                'facts': json.loads(row['facts']) if row['facts'] else [],
                'critical_info': json.loads(row['critical_info']) if row['critical_info'] else [],
                'difficulty_level': row['difficulty_level'],
                'harm_type': row['harm_type'],
                'expected_response': row['expected_response'] or '',
            }
        return scenarios

    def load_experiences(self) -> List[dict]:
        """Load all experiences."""
        experiences = []
        for row in self.conn.execute('SELECT * FROM experiences ORDER BY id'):
            experiences.append({
                'id': row['id'],
                'scenario_id': row['scenario_id'],
                'ai_response': row['ai_response'],
                'actual_harm': row['actual_harm'],
                'harm_breakdown': json.loads(row['harm_breakdown']) if row['harm_breakdown'] else {},
                'buddhist_scores': json.loads(row['buddhist_scores']) if row['buddhist_scores'] else {},
                'buddhist_alignment': row['buddhist_alignment'],
                'timestamp': row['timestamp'],
            })
        return experiences

    def process_experience(self, exp: dict, scenario: dict) -> Tuple[dict, CleaningReport]:
        """Clean and score a single experience."""
        # 1. Clean the response text
        cleaned_response, report = TextCleaner.clean(exp['ai_response'])
        report.experience_id = exp['id']
        report.scenario_id = exp['scenario_id']

        # 2. Calibration score
        calibration = CalibrationScorer.score(
            cleaned_response,
            scenario['context'],
            scenario['difficulty_level'],
            scenario['person_state'],
            scenario['expected_response']
        )
        report.calibration_score = calibration

        # 3. Adjusted Buddhist score
        original_weighted = sum(
            exp['buddhist_scores'].get(p, 5.0) * w
            for p, w in AdjustedBuddhistScorer.WEIGHTS.items()
        )
        adjusted_buddhist = AdjustedBuddhistScorer.adjusted_weighted_score(
            exp['buddhist_scores'],
            report.cleaned_word_count,
            calibration
        )
        report.original_buddhist_score = round(original_weighted, 3)
        report.adjusted_buddhist_score = adjusted_buddhist

        # 4. Coherence score
        coherence = CoherenceScorer.score(cleaned_response)

        # 5. Composite score for ranking
        composite = (
            adjusted_buddhist * 0.35 +
            calibration * 0.30 +
            coherence * 0.20 +
            (1.0 - exp['actual_harm']) * 10 * 0.15
        )

        processed = {
            **exp,
            'cleaned_response': cleaned_response,
            'harm_score': exp['actual_harm'],
            'calibration_score': calibration,
            'coherence_score': coherence,
            'adjusted_buddhist_score': adjusted_buddhist,
            'original_buddhist_score': original_weighted,
            'composite_score': round(composite, 3),
            'context': scenario['context'],
            'person_state': scenario['person_state'],
            'difficulty': scenario['difficulty_level'],
            'expected_response': scenario['expected_response'],
        }

        return processed, report

    def build_gold_examples(self, scenarios: Dict[int, dict]) -> List[dict]:
        """Build gold-standard examples from expected_response fields."""
        gold = []
        for sid, scenario in scenarios.items():
            if not scenario['expected_response']:
                continue
            response = scenario['expected_response']

            # Compute actual scores rather than assuming perfection
            calibration = CalibrationScorer.score(
                response,
                scenario['context'],
                scenario['difficulty_level'],
                scenario['person_state'],
                response,  # expected_response is itself the anchor
            )
            coherence = CoherenceScorer.score(response)
            # Buddhist scores aren't stored for gold; use a strong default
            adjusted_buddhist = 9.0
            composite = round(
                adjusted_buddhist * 0.35 +
                calibration * 0.30 +
                coherence * 0.20 +
                10.0 * 0.15,  # harm_score = 0.0
                3
            )

            gold.append({
                'cleaned_response': response,
                'context': scenario['context'],
                'person_state': scenario['person_state'],
                'difficulty': scenario['difficulty_level'],
                'harm_score': 0.0,
                'calibration_score': calibration,
                'coherence_score': coherence,
                'adjusted_buddhist_score': adjusted_buddhist,
                'composite_score': composite,
                'scenario_id': sid,
                'experience_id': f'gold_{sid}',
                'is_gold': True,
                'is_negative': False,
                'buddhist_alignment': 'excellent',
            })
        return gold

    def build_negative_examples(
        self, experiences: List[dict], scenarios: Dict[int, dict],
        max_harm: float = 1.0, min_harm: float = 0.2
    ) -> List[dict]:
        """
        Build negative examples from high-harm or low-calibration responses.
        These are useful for DPO training or contrastive learning.
        """
        negatives = []
        for exp in experiences:
            scenario = scenarios.get(exp['scenario_id'])
            if not scenario:
                continue

            is_negative = False
            reason = ''

            # High harm
            if exp['actual_harm'] >= min_harm:
                is_negative = True
                reason = f"high_harm={exp['actual_harm']:.2f}"

            # Extremely poor calibration on simple prompts
            if not is_negative:
                response_words = len(exp['ai_response'].split())
                prompt_type = CalibrationScorer.classify_prompt(
                    scenario['context'], scenario['difficulty_level'],
                    scenario['person_state']
                )
                if prompt_type == 'greeting' and response_words > 100:
                    is_negative = True
                    reason = f"greeting_verbosity={response_words}w"

            if is_negative:
                cleaned, _ = TextCleaner.clean(exp['ai_response'])

                # Compute actual scores for accurate diagnostics
                calibration = CalibrationScorer.score(
                    cleaned,
                    scenario['context'],
                    scenario['difficulty_level'],
                    scenario['person_state'],
                    scenario['expected_response'],
                )
                coherence = CoherenceScorer.score(cleaned)
                adjusted_buddhist = AdjustedBuddhistScorer.adjusted_weighted_score(
                    exp.get('buddhist_scores', {}),
                    len(cleaned.split()),
                    calibration,
                )
                composite = round(
                    adjusted_buddhist * 0.35 +
                    calibration * 0.30 +
                    coherence * 0.20 +
                    (1.0 - exp['actual_harm']) * 10 * 0.15,
                    3
                )

                negatives.append({
                    'cleaned_response': cleaned,
                    'context': scenario['context'],
                    'person_state': scenario['person_state'],
                    'difficulty': scenario['difficulty_level'],
                    'harm_score': exp['actual_harm'],
                    'calibration_score': calibration,
                    'coherence_score': coherence,
                    'adjusted_buddhist_score': adjusted_buddhist,
                    'composite_score': composite,
                    'scenario_id': exp['scenario_id'],
                    'experience_id': f"neg_{exp['id']}",
                    'is_gold': False,
                    'is_negative': True,
                    'buddhist_alignment': 'low',
                    'negative_reason': reason,
                })

        return negatives

    def convert(
        self,
        format_type: str = 'mistral',
        output_file: str = 'saige_training_data_v2.csv',
        max_harm: float = 0.25,
        min_composite: float = 5.0,
        min_calibration: float = 2.0,
        best_per_scenario: bool = False,
        include_gold: bool = False,
        include_negatives: bool = False,
        diagnostics_only: bool = False,
    ) -> Tuple[List[dict], List[CleaningReport]]:
        """Main conversion pipeline."""

        print("\n🧘 SAIGE v2 Training Data Pipeline")
        print("=" * 65)

        # Load data
        scenarios = self.load_scenarios()
        experiences = self.load_experiences()
        print(f"📦 Loaded {len(scenarios)} scenarios, {len(experiences)} experiences")

        # Process all experiences
        processed = []
        reports = []
        for exp in experiences:
            scenario = scenarios.get(exp['scenario_id'])
            if not scenario:
                continue
            p, r = self.process_experience(exp, scenario)
            processed.append(p)
            reports.append(r)

        print(f"🧹 Cleaned {len(processed)} experiences")

        # Pre-filter stats
        pre_filter = len(processed)

        # Apply quality filters
        filtered = []
        for p, r in zip(processed, reports):
            if p['harm_score'] > max_harm:
                r.kept = False
                r.rejection_reason = f"harm={p['harm_score']:.2f} > {max_harm}"
                continue
            if p['composite_score'] < min_composite:
                r.kept = False
                r.rejection_reason = f"composite={p['composite_score']:.2f} < {min_composite}"
                continue
            if p['calibration_score'] < min_calibration:
                r.kept = False
                r.rejection_reason = f"calibration={p['calibration_score']:.1f} < {min_calibration}"
                continue
            filtered.append(p)

        print(f"🔍 Filtered: {pre_filter} → {len(filtered)} examples")
        print(f"   Rejected: {pre_filter - len(filtered)} "
              f"(harm: {sum(1 for r in reports if 'harm' in r.rejection_reason)}, "
              f"composite: {sum(1 for r in reports if 'composite' in r.rejection_reason)}, "
              f"calibration: {sum(1 for r in reports if 'calibration' in r.rejection_reason)})")

        # Best per scenario (deduplication)
        if best_per_scenario:
            by_scenario = {}
            for p in filtered:
                sid = p['scenario_id']
                if sid not in by_scenario or p['composite_score'] > by_scenario[sid]['composite_score']:
                    by_scenario[sid] = p
            filtered = list(by_scenario.values())
            print(f"🏆 Best-per-scenario: {len(filtered)} examples")

        # Sort by composite score descending
        filtered.sort(key=lambda x: x['composite_score'], reverse=True)

        # Add gold standard examples
        final_examples = []
        if include_gold:
            gold = self.build_gold_examples(scenarios)
            final_examples.extend(gold)
            print(f"⭐ Added {len(gold)} gold-standard examples")

        # Add filtered positive examples
        for p in filtered:
            p['is_gold'] = False
            p['is_negative'] = False
            final_examples.append(p)

        # Add negative examples
        if include_negatives:
            negatives = self.build_negative_examples(experiences, scenarios)
            final_examples.extend(negatives)
            print(f"⚠️  Added {len(negatives)} negative examples")

        self.cleaning_reports = reports

        # Diagnostics
        if diagnostics_only:
            self._print_diagnostics(reports, processed, scenarios)
            return final_examples, reports

        # Format and write
        format_fn = self.FORMAT_MAP.get(format_type, FormatTemplates.mistral)
        self._write_csv(final_examples, format_fn, output_file)
        self._print_summary(final_examples, reports)
        self._write_diagnostics(reports, output_file)

        return final_examples, reports

    def _write_csv(self, examples: List[dict], format_fn, output_file: str):
        """Write training data CSV."""
        rows = []
        for ex in examples:
            # Build context note for vulnerable states
            context_note = ''
            ps = ex.get('person_state', {})
            mood = ps.get('mood', 'neutral')
            vuln = ps.get('vulnerability', 'low')
            if mood != 'neutral' or vuln != 'low':
                context_note = f"Person is {mood}, vulnerability: {vuln}"

            formatted = format_fn(
                ex['context'],
                ex['cleaned_response'],
                context_note
            )

            rows.append({
                'text': formatted,
                'harm_score': ex['harm_score'],
                'buddhist_alignment': ex.get('buddhist_alignment', ''),
                'weighted_score': ex.get('adjusted_buddhist_score', 0),
                'calibration_score': ex.get('calibration_score', 0),
                'coherence_score': ex.get('coherence_score', 0),
                'composite_score': ex.get('composite_score', 0),
                'difficulty': ex['difficulty'],
                'scenario_id': ex['scenario_id'],
                'experience_id': ex.get('experience_id', ''),
                'is_gold': ex.get('is_gold', False),
                'is_negative': ex.get('is_negative', False),
            })

        fieldnames = [
            'text', 'harm_score', 'buddhist_alignment', 'weighted_score',
            'calibration_score', 'coherence_score', 'composite_score',
            'difficulty', 'scenario_id', 'experience_id', 'is_gold', 'is_negative'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n💾 Wrote {len(rows)} examples to {output_file}")

    def _print_summary(self, examples: List[dict], reports: List[CleaningReport]):
        """Print summary statistics."""
        positives = [e for e in examples if not e.get('is_negative') and not e.get('is_gold')]
        golds = [e for e in examples if e.get('is_gold')]
        negatives = [e for e in examples if e.get('is_negative')]

        print(f"\n📊 v2 Training Data Summary")
        print(f"{'─' * 50}")
        print(f"  Total examples:     {len(examples)}")
        print(f"    Gold standard:    {len(golds)}")
        print(f"    Positive:         {len(positives)}")
        print(f"    Negative:         {len(negatives)}")

        if positives:
            avg_comp = sum(e['composite_score'] for e in positives) / len(positives)
            avg_cal = sum(e['calibration_score'] for e in positives) / len(positives)
            avg_coh = sum(e.get('coherence_score', 0) for e in positives) / len(positives)
            avg_bud = sum(e.get('adjusted_buddhist_score', 0) for e in positives) / len(positives)
            print(f"\n  Positive example metrics:")
            print(f"    Avg composite:    {avg_comp:.2f}/10")
            print(f"    Avg calibration:  {avg_cal:.2f}/10")
            print(f"    Avg coherence:    {avg_coh:.2f}/10")
            print(f"    Avg buddhist:     {avg_bud:.2f}/10")

        # Cleaning stats
        total_typos = sum(r.typos_fixed for r in reports)
        total_placeholders = sum(r.placeholders_removed for r in reports)
        total_signatures = sum(1 for r in reports if r.signatures_removed)
        total_prefixes = sum(r.ai_prefixes_removed for r in reports)
        avg_word_reduction = 0
        kept_reports = [r for r in reports if r.kept]
        if kept_reports:
            avg_word_reduction = sum(
                r.original_word_count - r.cleaned_word_count for r in kept_reports
            ) / len(kept_reports)

        print(f"\n  Cleaning performed:")
        print(f"    Typos fixed:      {total_typos}")
        print(f"    Placeholders:     {total_placeholders}")
        print(f"    Signatures:       {total_signatures}")
        print(f"    AI: prefixes:     {total_prefixes}")
        print(f"    Avg words removed: {avg_word_reduction:.0f} per example")

        # Difficulty distribution
        diff_counts = {}
        for e in positives:
            d = e['difficulty']
            diff_counts[d] = diff_counts.get(d, 0) + 1
        print(f"\n  Difficulty distribution (positives):")
        for d in sorted(diff_counts.keys()):
            c = diff_counts[d]
            print(f"    Level {d}: {c} ({c / len(positives) * 100:.0f}%)")

    def _print_diagnostics(self, reports, processed, scenarios):
        """Print detailed diagnostics for analysis."""
        print(f"\n🔬 Detailed Diagnostics")
        print(f"{'═' * 65}")

        # Worst calibration examples
        print(f"\n{'─' * 65}")
        print("WORST CALIBRATION (responses too long for their prompt):")
        print(f"{'─' * 65}")
        by_cal = sorted(processed, key=lambda x: x['calibration_score'])
        for p in by_cal[:10]:
            s = scenarios[p['scenario_id']]
            eid = p.get('experience_id', p.get('id', '?'))
            print(f"  [{eid}] Scenario: \"{s['context'][:40]}\"")
            print(f"     Calibration: {p['calibration_score']:.1f}/10  |  "
                  f"Response: {len(p['cleaned_response'].split())}w  |  "
                  f"Expected: ~{len(s['expected_response'].split())}w")
            print()

        # Most improved by cleaning
        print(f"{'─' * 65}")
        print("MOST WORDS REMOVED BY CLEANING:")
        print(f"{'─' * 65}")
        by_reduction = sorted(reports, key=lambda r: r.original_word_count - r.cleaned_word_count, reverse=True)
        for r in by_reduction[:5]:
            delta = r.original_word_count - r.cleaned_word_count
            print(f"  exp_id={r.experience_id}: {r.original_word_count}w → {r.cleaned_word_count}w (−{delta}w)")
            print(f"     typos={r.typos_fixed}, placeholders={r.placeholders_removed}, "
                  f"sigs={r.signatures_removed}, ai_prefix={r.ai_prefixes_removed}")
            print()

        # Rejection reasons
        print(f"{'─' * 65}")
        print("REJECTION BREAKDOWN:")
        print(f"{'─' * 65}")
        rejected = [r for r in reports if not r.kept]
        reasons = {}
        for r in rejected:
            key = r.rejection_reason.split('=')[0] if '=' in r.rejection_reason else r.rejection_reason
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    def _write_diagnostics(self, reports: List[CleaningReport], base_output: str):
        """Write cleaning diagnostics to a companion file."""
        diag_file = base_output.replace('.csv', '_diagnostics.csv')
        fieldnames = [
            'experience_id', 'scenario_id', 'original_word_count', 'cleaned_word_count',
            'typos_fixed', 'placeholders_removed', 'signatures_removed',
            'ai_prefixes_removed', 'calibration_score', 'original_buddhist_score',
            'adjusted_buddhist_score', 'kept', 'rejection_reason'
        ]

        with open(diag_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in reports:
                writer.writerow({
                    'experience_id': r.experience_id,
                    'scenario_id': r.scenario_id,
                    'original_word_count': r.original_word_count,
                    'cleaned_word_count': r.cleaned_word_count,
                    'typos_fixed': r.typos_fixed,
                    'placeholders_removed': r.placeholders_removed,
                    'signatures_removed': r.signatures_removed,
                    'ai_prefixes_removed': r.ai_prefixes_removed,
                    'calibration_score': r.calibration_score,
                    'original_buddhist_score': r.original_buddhist_score,
                    'adjusted_buddhist_score': r.adjusted_buddhist_score,
                    'kept': r.kept,
                    'rejection_reason': r.rejection_reason,
                })

        print(f"📋 Diagnostics written to {diag_file}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SAIGE v2 — Training Data Cleaner & Improver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: clean + filter + output Mistral format
  python saige_to_sft_v2.py --db saige.db

  # Best single example per scenario + gold standards
  python saige_to_sft_v2.py --db saige.db --best-per-scenario --include-gold

  # Include negatives for DPO/contrastive training
  python saige_to_sft_v2.py --db saige.db --include-negatives --include-gold

  # ChatML format for TinyLlama
  python saige_to_sft_v2.py --db saige.db --format chatml

  # Diagnostics only (no output file)
  python saige_to_sft_v2.py --db saige.db --diagnostics-only

  # Strict filtering
  python saige_to_sft_v2.py --db saige.db --min-composite 7.0 --min-calibration 5.0
        """
    )

    parser.add_argument('--db', default='saige.db', help='Path to SAIGE database')
    parser.add_argument('--output', default='saige_training_data_v2.csv', help='Output CSV file')
    parser.add_argument('--format', choices=['mistral', 'chatml', 'llama3', 'alpaca'],
                        default='mistral', help='Training format (default: mistral)')
    parser.add_argument('--max-harm', type=float, default=0.25,
                        help='Maximum harm score (default: 0.25)')
    parser.add_argument('--min-composite', type=float, default=5.0,
                        help='Minimum composite quality score (default: 5.0)')
    parser.add_argument('--min-calibration', type=float, default=2.0,
                        help='Minimum calibration score (default: 2.0)')
    parser.add_argument('--best-per-scenario', action='store_true',
                        help='Keep only the best example per scenario')
    parser.add_argument('--include-gold', action='store_true',
                        help='Include gold-standard expected responses')
    parser.add_argument('--include-negatives', action='store_true',
                        help='Include negative examples (for DPO/contrastive)')
    parser.add_argument('--diagnostics-only', action='store_true',
                        help='Print diagnostics without writing output')

    args = parser.parse_args()

    converter = SAIGEv2Converter(args.db)
    try:
        converter.connect()
        converter.convert(
            format_type=args.format,
            output_file=args.output,
            max_harm=args.max_harm,
            min_composite=args.min_composite,
            min_calibration=args.min_calibration,
            best_per_scenario=args.best_per_scenario,
            include_gold=args.include_gold,
            include_negatives=args.include_negatives,
            diagnostics_only=args.diagnostics_only,
        )
    finally:
        converter.disconnect()


if __name__ == '__main__':
    main()
