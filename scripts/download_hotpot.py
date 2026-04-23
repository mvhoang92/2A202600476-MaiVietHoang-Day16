"""
Script tải dataset HotpotQA và convert sang format QAExample.

Cách chạy:
    pip install datasets
    python scripts/download_hotpot.py

Output: data/hotpot_100.json (150 mẫu đầu từ validation split)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Thêm src vào path để import schemas
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexion_lab.schemas import QAExample, ContextChunk

NUM_SAMPLES = 150
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "hotpot_100.json"
MAX_CONTEXT_CHUNKS = 4


def convert_item(item: dict, index: int) -> dict:
    """Convert một item HotpotQA sang format QAExample."""
    titles = item["context"]["title"]
    sentences_list = item["context"]["sentences"]

    context_chunks = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        context_chunks.append({"title": title, "text": text})

    # Giới hạn 4 chunks đầu
    context_chunks = context_chunks[:MAX_CONTEXT_CHUNKS]

    return {
        "qid": f"hpqa_{index:04d}",
        "difficulty": "medium",
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context_chunks,
    }


def download_via_huggingface() -> list[dict]:
    """Tải dataset qua HuggingFace datasets library."""
    from datasets import load_dataset  # type: ignore

    print("Đang tải hotpot_qa (distractor, validation) từ HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= NUM_SAMPLES:
            break
        samples.append(convert_item(item, i))

    return samples


def download_via_url() -> list[dict]:
    """Fallback: tải trực tiếp từ URL công khai."""
    import gzip
    import urllib.request

    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    print(f"Đang tải từ URL: {url}")

    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()

    # Thử giải nén nếu là gzip
    try:
        data = json.loads(gzip.decompress(raw))
    except Exception:
        data = json.loads(raw)

    samples = []
    for i, item in enumerate(data):
        if i >= NUM_SAMPLES:
            break

        # Format từ file JSON gốc của HotpotQA
        titles = [ctx[0] for ctx in item.get("context", [])]
        sentences_list = [ctx[1] for ctx in item.get("context", [])]

        context_chunks = []
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            context_chunks.append({"title": title, "text": text})

        context_chunks = context_chunks[:MAX_CONTEXT_CHUNKS]

        samples.append({
            "qid": f"hpqa_{i:04d}",
            "difficulty": "medium",
            "question": item["question"],
            "gold_answer": item["answer"],
            "context": context_chunks,
        })

    return samples


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples = None

    # Thử HuggingFace datasets trước
    try:
        samples = download_via_huggingface()
    except ImportError:
        print("Thư viện 'datasets' chưa được cài. Thử fallback URL...")
    except Exception as e:
        print(f"HuggingFace load thất bại: {e}. Thử fallback URL...")

    # Fallback: tải từ URL
    if samples is None:
        try:
            samples = download_via_url()
        except Exception as e:
            print(f"Fallback URL cũng thất bại: {e}")
            print("\nVui lòng chạy thủ công:")
            print("  pip install datasets")
            print("  python scripts/download_hotpot.py")
            sys.exit(1)

    # Validate qua Pydantic
    validated = [QAExample.model_validate(s).model_dump() for s in samples]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu {len(validated)} mẫu vào {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
