from sentence_transformers import SentenceTransformer
import torch
import os
import gc
import json
from .utils import get_md5


def _get_embedding_cache_dir():
    """Return a stable cache directory for tool embeddings.

    Uses ``~/.cache/txagent/embeddings`` so that the cached ``.pt`` file
    is found regardless of the working directory at launch time.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "txagent", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class ToolRAGModel:
    def __init__(self, rag_model_name):
        self.rag_model_name = rag_model_name
        self.rag_model = None
        self.tool_desc_embedding = None
        self.tool_name = None
        self.tool_embedding_path = None
        self.load_rag_model()

    def load_rag_model(self):
        self.rag_model = SentenceTransformer(self.rag_model_name)
        self.rag_model.max_seq_length = 4096
        self.rag_model.tokenizer.padding_side = "right"

    def load_tool_desc_embedding(self, toolbox):
        self.tool_name, _ = toolbox.refresh_tool_name_desc(
            enable_full_desc=True)
        all_tools_str = [json.dumps(
            each) for each in toolbox.prepare_tool_prompts(toolbox.all_tools)]
        md5_value = get_md5(str(all_tools_str))
        print("get the md value of tools:", md5_value)

        embedding_filename = self.rag_model_name.split(
            '/')[-1] + "tool_embedding_" + md5_value + ".pt"

        cache_dir = _get_embedding_cache_dir()
        self.tool_embedding_path = os.path.join(cache_dir, embedding_filename)

        # Also check legacy relative-path location (backward compat)
        legacy_path = os.path.join(os.getcwd(), embedding_filename)
        if not os.path.exists(self.tool_embedding_path) and os.path.exists(legacy_path):
            print(f"Migrating legacy embedding cache from {legacy_path} to {self.tool_embedding_path}")
            os.rename(legacy_path, self.tool_embedding_path)

        try:
            self.tool_desc_embedding = torch.load(
                self.tool_embedding_path, weights_only=False)
            assert len(self.tool_desc_embedding) == len(
                toolbox.all_tools), (
                f"The number of tools in the toolbox ({len(toolbox.all_tools)}) "
                f"does not match the cached embeddings ({len(self.tool_desc_embedding)}). "
                "Regenerating embeddings."
            )
        except FileNotFoundError:
            print(f"\033[93mEmbedding cache not found at {self.tool_embedding_path}. "
                  "Generating tool description embeddings (this may take a few minutes).\033[0m")
            self._generate_and_cache_embeddings(all_tools_str)
        except AssertionError as e:
            print(f"\033[93m{e}\033[0m")
            print("Regenerating tool description embeddings...")
            self._generate_and_cache_embeddings(all_tools_str)

    def _generate_and_cache_embeddings(self, all_tools_str):
        """Generate embeddings for all tool descriptions and cache to disk.

        After saving, frees GPU memory used by the encoder so the main
        model can continue without OOM.
        """
        self.tool_desc_embedding = self.rag_model.encode(
            all_tools_str, prompt="", normalize_embeddings=True
        )
        torch.save(self.tool_desc_embedding, self.tool_embedding_path)
        print(f"\033[92mTool embeddings saved to {self.tool_embedding_path}\033[0m")

        # Free encoder GPU memory so the main LLM can continue
        del self.rag_model
        self.rag_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\033[92mFreed RAG encoder GPU memory. Re-loading RAG model...\033[0m")
        self.load_rag_model()
        # Reload the cached embeddings (they may have been freed)
        self.tool_desc_embedding = torch.load(
            self.tool_embedding_path, weights_only=False)

    def rag_infer(self, query, top_k=5):
        torch.cuda.empty_cache()
        queries = [query]
        query_embeddings = self.rag_model.encode(
            queries, prompt="", normalize_embeddings=True
        )
        if self.tool_desc_embedding is None:
            print("No tool_desc_embedding")
            exit()
        scores = self.rag_model.similarity(
            query_embeddings, self.tool_desc_embedding)
        top_k = min(top_k, len(self.tool_name))
        top_k_indices = torch.topk(scores, top_k).indices.tolist()[0]
        top_k_tool_names = [self.tool_name[i] for i in top_k_indices]
        return top_k_tool_names
