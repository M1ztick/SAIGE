"""
Upload DPO pairs to M1ztyk/SAIGE-right-speech-dpo on HuggingFace Hub.

Run from project root:
  python3 local-trainer/upload_dataset.py
"""

import json
from pathlib import Path
from huggingface_hub import HfApi

JSONL_PATH = Path(__file__).parent / "dpo_pairs.jsonl"
REPO_ID = "M1ztyk/SAIGE-right-speech-dpo"

KEEP_FIELDS = {"prompt", "chosen", "rejected", "record_id", "path_factor", "pair_type", "score_delta", "prompt_type"}

def main():
    raw = [json.loads(l) for l in JSONL_PATH.read_text().splitlines() if l.strip()]
    cleaned = [{k: v for k, v in pair.items() if k in KEEP_FIELDS} for pair in raw]

    out_path = JSONL_PATH.parent / "dpo_pairs_upload.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in cleaned))

    print(f"Prepared {len(cleaned)} pairs → {out_path.name}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="dpo_pairs.jsonl",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {len(cleaned)} DPO pairs (Right Speech track, rs-001–rs-012)",
    )

    out_path.unlink()
    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
