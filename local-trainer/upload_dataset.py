"""
Upload prompt-diversified DPO pairs to M1ztyk/SAIGE-right-speech-dpo on HuggingFace Hub.

Requires the hf CLI to be installed and authenticated:
  hf auth login

Run from project root:
  python3 local-trainer/upload_dataset.py
"""

import json
import subprocess
from pathlib import Path

JSONL_PATH = Path(__file__).parent / "dpo_pairs_diversified.jsonl"
REPO_ID = "M1ztyk/SAIGE-right-speech-dpo"

KEEP_FIELDS = {
    "prompt", "chosen", "rejected", "record_id",
    "path_factor", "pair_type", "score_delta", "prompt_type",
    "prompt_condition",
}


def main():
    """Clean and upload the diversified DPO pairs to HuggingFace Hub via hf CLI."""
    raw = [json.loads(l) for l in JSONL_PATH.read_text().splitlines() if l.strip()]
    cleaned = [{k: v for k, v in pair.items() if k in KEEP_FIELDS} for pair in raw]

    out_path = JSONL_PATH.parent / "dpo_pairs_upload.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in cleaned))
    print(f"Prepared {len(cleaned)} pairs → {out_path.name}")

    hf_cli = Path.home() / ".local/bin/hf"
    result = subprocess.run(
        [
            str(hf_cli), "upload", REPO_ID,
            str(out_path), "dpo_pairs.jsonl",
            "--repo-type", "dataset",
            "--commit-message",
            f"Add {len(cleaned)} prompt-diversified pairs (rs/generic/none conditions)",
        ],
        capture_output=True,
        text=True,
    )

    out_path.unlink()

    if result.returncode != 0:
        print(f"Upload failed:\n{result.stderr}")
    else:
        print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
