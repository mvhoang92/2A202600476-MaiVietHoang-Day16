"""Integration tests cho dataset HotpotQA."""
from __future__ import annotations

import pytest
from pathlib import Path

HOTPOT_PATH = Path("data/hotpot_100.json")


@pytest.mark.skipif(not HOTPOT_PATH.exists(), reason="data/hotpot_100.json chưa được tạo")
def test_hotpot_dataset_loads_at_least_100_samples():
    """load_dataset trả về ≥100 mẫu hợp lệ."""
    from src.reflexion_lab.utils import load_dataset

    examples = load_dataset(HOTPOT_PATH)
    assert len(examples) >= 100, f"Chỉ có {len(examples)} mẫu, cần ≥100"


@pytest.mark.skipif(not HOTPOT_PATH.exists(), reason="data/hotpot_100.json chưa được tạo")
def test_hotpot_dataset_sample_fields():
    """Mỗi mẫu có đủ các trường qid, question, gold_answer, context."""
    from src.reflexion_lab.utils import load_dataset

    examples = load_dataset(HOTPOT_PATH)
    for ex in examples:
        assert ex.qid.startswith("hpqa_"), f"qid không đúng format: {ex.qid}"
        assert ex.question, "question không được rỗng"
        assert ex.gold_answer, "gold_answer không được rỗng"
        assert isinstance(ex.context, list), "context phải là list"
        assert len(ex.context) > 0, "context không được rỗng"
        assert len(ex.context) <= 4, f"context vượt quá 4 chunks: {len(ex.context)}"


@pytest.mark.skipif(not HOTPOT_PATH.exists(), reason="data/hotpot_100.json chưa được tạo")
def test_hotpot_dataset_context_chunks_have_title_and_text():
    """Mỗi context chunk có title và text."""
    from src.reflexion_lab.utils import load_dataset

    examples = load_dataset(HOTPOT_PATH)
    for ex in examples[:10]:  # Kiểm tra 10 mẫu đầu
        for chunk in ex.context:
            assert chunk.title, "title không được rỗng"
            assert chunk.text, "text không được rỗng"
