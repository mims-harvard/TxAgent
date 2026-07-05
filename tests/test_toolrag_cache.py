from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txagent.toolrag import ToolRAGModel


def test_toolrag_uses_configured_cache_dir(tmp_path):
    with patch.object(ToolRAGModel, "load_rag_model"):
        model = ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B", cache_dir=tmp_path)

    assert model.cache_dir == Path(tmp_path)
    assert model.rag_model_name == "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
