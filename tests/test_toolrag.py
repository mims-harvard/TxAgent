"""Tests for the improved tool embedding cache in toolrag.py.

Verifies that:
1. Missing embedding files trigger generation (not an unhandled crash)
2. Cached embeddings are loaded correctly
3. Mismatched tool counts trigger regeneration
4. Legacy relative-path embeddings are migrated
5. GPU memory is freed after generation
"""
import os
import sys
import json
import tempfile
import shutil
import hashlib
from unittest import mock

import torch
import numpy as np

# We need to import toolrag directly without going through __init__.py
# (which imports txagent.py -> vllm which isn't available on macOS)
# So we mock the heavy dependencies and import the module directly.

# Mock sentence_transformers so we don't need the real model
fake_st = mock.MagicMock()

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_fake_toolbox(n_tools=5):
    """Return a mock toolbox with n_tools fake tools."""
    toolbox = mock.MagicMock()
    tools = [{"name": f"tool_{i}", "desc": f"description_{i}"} for i in range(n_tools)]
    toolbox.all_tools = tools
    toolbox.refresh_tool_name_desc.return_value = ([f"tool_{i}" for i in range(n_tools)], None)
    toolbox.prepare_tool_prompts.return_value = tools
    return toolbox


def _make_fake_sentence_transformer():
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    model = mock.MagicMock()
    def fake_encode(texts, prompt="", normalize_embeddings=True):
        return np.random.randn(len(texts), 32).astype(np.float32)
    model.encode.side_effect = fake_encode
    def fake_similarity(a, b):
        return torch.randn(a.shape[0], b.shape[0])
    model.similarity.side_effect = fake_similarity
    return model


def _import_toolrag():
    """Import toolrag module with sentence_transformers mocked."""
    with mock.patch.dict('sys.modules', {'sentence_transformers': fake_st}):
        # Clear cached module if present
        if 'txagent.toolrag' in sys.modules:
            del sys.modules['txagent.toolrag']
        if 'txagent' in sys.modules:
            del sys.modules['txagent']
        if 'txagent.utils' in sys.modules:
            del sys.modules['txagent.utils']

        # Mock the relative import of .utils
        utils_mock = mock.MagicMock()
        def real_get_md5(input_str):
            md5_hash = hashlib.md5()
            md5_hash.update(input_str.encode('utf-8'))
            return md5_hash.hexdigest()
        utils_mock.get_md5 = real_get_md5
        sys.modules['txagent'] = mock.MagicMock()
        sys.modules['txagent.utils'] = utils_mock

        # Now import from source
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

        spec = importlib.util.spec_from_file_location(
            "txagent.toolrag",
            os.path.join(os.path.dirname(__file__), "..", "src", "txagent", "toolrag.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules['txagent.toolrag'] = mod
        spec.loader.exec_module(mod)
        return mod


# ── tests ────────────────────────────────────────────────────────────────────

def test_missing_file_generates_embeddings():
    """When no cached .pt file exists, embeddings should be generated and saved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = _import_toolrag()
        with mock.patch.object(mod, "_get_embedding_cache_dir", return_value=tmpdir):
            with mock.patch.object(mod, "SentenceTransformer", return_value=_make_fake_sentence_transformer()):
                model = mod.ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")

                toolbox = _make_fake_toolbox(5)
                model.load_tool_desc_embedding(toolbox)

                assert model.tool_desc_embedding is not None
                assert os.path.exists(model.tool_embedding_path)
                assert len(model.tool_desc_embedding) == 5
                print("PASS: test_missing_file_generates_embeddings")


def test_cached_file_is_loaded():
    """When a valid cached .pt file exists, it should be loaded without regeneration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = _import_toolrag()
        with mock.patch.object(mod, "_get_embedding_cache_dir", return_value=tmpdir):
            with mock.patch.object(mod, "SentenceTransformer", return_value=_make_fake_sentence_transformer()):
                model = mod.ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")

                toolbox = _make_fake_toolbox(5)

                # Pre-create a valid cache file
                all_tools_str = [json.dumps(t) for t in toolbox.prepare_tool_prompts(toolbox.all_tools)]
                md5_value = hashlib.md5(str(all_tools_str).encode()).hexdigest()
                embedding_filename = "ToolRAG-T1-GTE-Qwen2-1.5Btool_embedding_" + md5_value + ".pt"
                cached_path = os.path.join(tmpdir, embedding_filename)
                fake_embeddings = np.random.randn(5, 32).astype(np.float32)
                torch.save(fake_embeddings, cached_path)

                model.load_tool_desc_embedding(toolbox)

                np.testing.assert_array_equal(model.tool_desc_embedding, fake_embeddings)
                print("PASS: test_cached_file_is_loaded")


def test_mismatched_tool_count_regenerates():
    """When cached embeddings have wrong count, they should be regenerated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = _import_toolrag()
        with mock.patch.object(mod, "_get_embedding_cache_dir", return_value=tmpdir):
            with mock.patch.object(mod, "SentenceTransformer", return_value=_make_fake_sentence_transformer()):
                model = mod.ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")

                toolbox = _make_fake_toolbox(5)

                # Pre-create cache with wrong count (3 instead of 5)
                all_tools_str = [json.dumps(t) for t in toolbox.prepare_tool_prompts(toolbox.all_tools)]
                md5_value = hashlib.md5(str(all_tools_str).encode()).hexdigest()
                embedding_filename = "ToolRAG-T1-GTE-Qwen2-1.5Btool_embedding_" + md5_value + ".pt"
                cached_path = os.path.join(tmpdir, embedding_filename)
                wrong_embeddings = np.random.randn(3, 32).astype(np.float32)
                torch.save(wrong_embeddings, cached_path)

                model.load_tool_desc_embedding(toolbox)

                assert len(model.tool_desc_embedding) == 5
                print("PASS: test_mismatched_tool_count_regenerates")


def test_legacy_path_migration():
    """Legacy relative-path embeddings should be migrated to the cache dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        cwd_dir = os.path.join(tmpdir, "cwd")
        os.makedirs(cwd_dir)

        mod = _import_toolrag()
        with mock.patch.object(mod, "_get_embedding_cache_dir", return_value=cache_dir):
            with mock.patch.object(mod, "SentenceTransformer", return_value=_make_fake_sentence_transformer()):
                with mock.patch("os.getcwd", return_value=cwd_dir):
                    model = mod.ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")

                    toolbox = _make_fake_toolbox(5)

                    all_tools_str = [json.dumps(t) for t in toolbox.prepare_tool_prompts(toolbox.all_tools)]
                    md5_value = hashlib.md5(str(all_tools_str).encode()).hexdigest()
                    embedding_filename = "ToolRAG-T1-GTE-Qwen2-1.5Btool_embedding_" + md5_value + ".pt"
                    legacy_path = os.path.join(cwd_dir, embedding_filename)
                    fake_embeddings = np.random.randn(5, 32).astype(np.float32)
                    torch.save(fake_embeddings, legacy_path)

                    model.load_tool_desc_embedding(toolbox)

                    assert not os.path.exists(legacy_path), "Legacy file should be moved"
                    assert os.path.exists(model.tool_embedding_path), "File should be in cache dir"
                    np.testing.assert_array_equal(model.tool_desc_embedding, fake_embeddings)
                    print("PASS: test_legacy_path_migration")


def test_gpu_memory_freed_after_generation():
    """After generating embeddings, the RAG model should be deleted and re-loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = _import_toolrag()
        with mock.patch.object(mod, "_get_embedding_cache_dir", return_value=tmpdir):
            call_count = [0]
            def make_model(_name=None):
                call_count[0] += 1
                return _make_fake_sentence_transformer()
            with mock.patch.object(mod, "SentenceTransformer", side_effect=make_model):
                with mock.patch("gc.collect") as mock_gc:
                    model = mod.ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")

                    toolbox = _make_fake_toolbox(5)
                    initial_calls = call_count[0]
                    model.load_tool_desc_embedding(toolbox)

                    assert call_count[0] > initial_calls, "RAG model should be re-loaded after generation"
                    assert mock_gc.called, "gc.collect should be called to free memory"
                    print("PASS: test_gpu_memory_freed_after_generation")


def test_no_exit_call_on_first_run():
    """The old code called exit() after generating embeddings — the fix should NOT do that."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = _import_toolrag()
        with mock.patch.object(mod, "_get_embedding_cache_dir", return_value=tmpdir):
            with mock.patch.object(mod, "SentenceTransformer", return_value=_make_fake_sentence_transformer()):
                model = mod.ToolRAGModel("mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")
                toolbox = _make_fake_toolbox(5)

                # This should NOT raise SystemExit
                model.load_tool_desc_embedding(toolbox)
                # If we get here, exit() was not called
                assert model.tool_desc_embedding is not None
                print("PASS: test_no_exit_call_on_first_run")


if __name__ == "__main__":
    test_missing_file_generates_embeddings()
    test_cached_file_is_loaded()
    test_mismatched_tool_count_regenerates()
    test_legacy_path_migration()
    test_gpu_memory_freed_after_generation()
    test_no_exit_call_on_first_run()
    print("\n=== ALL TESTS PASSED ===")
