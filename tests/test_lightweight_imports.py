from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_package_import_does_not_eagerly_load_runtime_dependencies():
    import txagent

    assert txagent.__all__ == ["TxAgent", "ToolRAGModel"]
