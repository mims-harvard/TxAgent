import torch
import json
from .utils import get_md5


class ToolRAGModel:
    def __init__(self, rag_model_name):
        self.rag_model_name = rag_model_name
        self.rag_model = None
        self.tool_desc_embedding = None
        self.tool_name = None
        self.tool_embedding_path = None

    def load_rag_model(self):
        if self.rag_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for ToolRAGModel. "
                "Install it with `pip install sentence-transformers` and retry."
            ) from exc

        self.rag_model = SentenceTransformer(self.rag_model_name)
        self.rag_model.max_seq_length = 4096
        self.rag_model.tokenizer.padding_side = "right"

    def load_tool_desc_embedding(self, toolbox):
        self.load_rag_model()
        self.tool_name, _ = toolbox.refresh_tool_name_desc(
            enable_full_desc=True)
        all_tools_str = [json.dumps(
            each) for each in toolbox.prepare_tool_prompts(toolbox.all_tools)]
        md5_value = get_md5(str(all_tools_str))
        print("get the md value of tools:", md5_value)
        self.tool_embedding_path = self.rag_model_name.split(
            '/')[-1] + "tool_embedding_" + md5_value + ".pt"
        try:
            self.tool_desc_embedding = torch.load(
                self.tool_embedding_path, weights_only=False)
            assert len(self.tool_desc_embedding) == len(
                toolbox.all_tools), "The number of tools in the toolbox is not equal to the number of tool_desc_embedding."
        except:
            self.tool_desc_embedding = None
            print("\033[92mInferring the tool_desc_embedding.\033[0m")
            self.tool_desc_embedding = self.rag_model.encode(
                all_tools_str, prompt="", normalize_embeddings=True
            )
            torch.save(self.tool_desc_embedding, self.tool_embedding_path)
            print("\033[92mFinished inferring the tool_desc_embedding.\033[0m")
            print("\033[91mExiting. Please rerun the code to avoid the OOM issue.\033[0m")
            exit()

    def rag_infer(self, query, top_k=5):
        self.load_rag_model()
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
