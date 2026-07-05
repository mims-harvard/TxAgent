from sentence_transformers import SentenceTransformer
import torch
import json
import os
from pathlib import Path
from .utils import get_md5


class ToolRAGModel:
    def __init__(self, rag_model_name, cache_dir=None):
        self.rag_model_name = rag_model_name
        self.cache_dir = Path(
            cache_dir or os.environ.get("TXAGENT_TOOLRAG_CACHE_DIR", ".")
        )
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_slug = self.rag_model_name.rstrip("/").split("/")[-1]
        self.tool_embedding_path = str(
            self.cache_dir / f"{model_slug}_tool_embedding_{md5_value}.pt"
        )
        try:
            self.tool_desc_embedding = torch.load(
                self.tool_embedding_path, weights_only=False)
            assert len(self.tool_desc_embedding) == len(
                toolbox.all_tools), "The number of tools in the toolbox is not equal to the number of tool_desc_embedding."
        except Exception as exc:
            self.tool_desc_embedding = None
            print(f"Could not load ToolRAG embedding cache: {exc}")
            print("\033[92mInferring the tool_desc_embedding.\033[0m")
            self.tool_desc_embedding = self.rag_model.encode(
                all_tools_str, prompt="", normalize_embeddings=True
            )
            torch.save(self.tool_desc_embedding, self.tool_embedding_path)
            print("\033[92mFinished inferring the tool_desc_embedding.\033[0m")
            print("\033[91mExiting. Please rerun the code to avoid the OOM issue.\033[0m")
            exit()

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
