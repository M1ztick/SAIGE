#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import pandas as pd


TOKEN_PATTERNS = [
    r"<s>",
    r"</s>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|endoftext\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]


def normalize_bool(val):
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y"}


def strip_template_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""

    cleaned = text

    for pattern in TOKEN_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Remove duplicated role labels at line starts
    cleaned = re.sub(r"(?im)^\s*(assistant|ai assistant|ai|user|human|system)\s*:\s*", "", cleaned)

    # Collapse whitespace
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def split_prompt_response(text: str):
    """
    Handles patterns like:
      <s>[INST] prompt [/INST] response </s>
    and a few fallback formats.
    """
    if not isinstance(text, str):
        return "", ""

    raw = text.strip()

    # Best case: classic Mistral/Llama instruct format
    m = re.search(r"\[INST\](.*?)\[/INST\](.*)", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        prompt = strip_template_tokens(m.group(1))
        response = strip_template_tokens(m.group(2))
        return prompt, response

    # ChatML-ish fallback: split on assistant marker
    m = re.search(
        r"(.*?)(?:<\|assistant\|>|(?:^|\n)\s*assistant\s*:)(.*)",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        prompt = strip_template_tokens(m.group(1))
        response = strip_template_tokens(m.group(2))
        return prompt, response

    # Human/Assistant fallback
    m = re.search(
        r"(.*?)(?:^|\n)\s*(?:assistant|ai assistant|ai)\s*:\s*(.*)",
        raw,
        flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    if m:
        prompt = strip_template_tokens(m.group(1))
        response = strip_template_tokens(m.group(2))
        return prompt, response

    # No clean split found: treat the whole thing as response-only junk
    return "", strip_template_tokens(raw)


def main():
    parser = argparse.ArgumentParser(description="Clean SAIGE CSV into gold-only prompt/response pairs.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="saige_sft_gold.csv", help="Output CSV path")
    parser.add_argument(
        "--rejected-output",
        default="saige_rejected.csv",
        help="Rejected / unusable rows output CSV path"
    )
    parser.add_argument("--min-response-chars", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    if "is_gold" not in df.columns:
        print("Warning: no 'is_gold' column found. Assuming all rows are gold.")
        df["is_gold"] = True

    df["is_gold"] = df["is_gold"].apply(normalize_bool)

    gold_df = df[df["is_gold"] == True].copy()
    non_gold_df = df[df["is_gold"] != True].copy()

    print(f"Gold rows: {len(gold_df)}")
    print(f"Non-gold rows: {len(non_gold_df)}")

    cleaned_rows = []
    rejected_rows = []

    for idx, row in gold_df.iterrows():
        text = row.get("text", "")
        prompt, response = split_prompt_response(text)

        if not prompt or not response or len(response.strip()) < args.min_response_chars:
            rejected_rows.append({
                "row_index": idx,
                "reason": "could_not_split_or_response_too_short",
                "original_text": text,
            })
            continue

        cleaned_rows.append({
            "prompt": prompt,
            "response": response,
        })

    # Deduplicate exact prompt/response pairs
    out_df = pd.DataFrame(cleaned_rows).drop_duplicates(subset=["prompt", "response"]).reset_index(drop=True)
    rej_df = pd.DataFrame(rejected_rows)

    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    rej_df.to_csv(args.rejected_output, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nWrote {len(out_df)} cleaned gold rows to: {args.output}")
    print(f"Wrote {len(rej_df)} rejected gold rows to: {args.rejected_output}")

    if len(out_df) > 0:
        print("\nSample cleaned row:")
        print("-" * 80)
        print("PROMPT:")
        print(out_df.iloc[0]['prompt'][:500])
        print("\nRESPONSE:")
        print(out_df.iloc[0]['response'][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
