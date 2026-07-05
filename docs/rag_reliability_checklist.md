# RAG Reliability Checklist

Use this checklist when TxAgent returns a shallow answer, skips expected tools,
or cites evidence that does not match the user question.

## Setup

- Confirm ToolUniverse loads before running the agent.
- Confirm the ToolRAG embedding cache was generated for the current
  ToolUniverse tool set.
- Set `TXAGENT_TOOLRAG_CACHE_DIR` to a writable directory when the working
  directory is read-only or shared across runs.
- Record `init_rag_num`, `step_rag_num`, `call_agent`, model name, and vLLM
  settings with each reproduction.
- Re-run with the same prompt and fixed seed before changing multiple settings.

## ToolRAG Cache

The cache filename is derived from the RAG model name and the MD5 of the current
ToolUniverse tool descriptions. If tools are added, removed, or their
descriptions change, TxAgent creates a new embedding cache.

```bash
export TXAGENT_TOOLRAG_CACHE_DIR=/path/to/txagent-toolrag-cache
```

Keep the cache directory on fast local storage for repeated experiments. Delete
only the matching `*_tool_embedding_*.pt` file when you intentionally want to
rebuild embeddings for a changed tool set.

## Query and Tool Selection

- Check whether the prompt names the drug, disease, patient attribute, or
  mechanism needed for retrieval.
- Increase `init_rag_num` when no relevant tools appear in the initial tool
  context.
- Increase `step_rag_num` when the agent starts correctly but cannot retrieve
  enough follow-up tools after intermediate observations.
- Use `call_agent=True` when you want the agentic tool-calling loop rather than
  a direct model response.

## Evidence Grounding

- Compare each final recommendation with the tool outputs returned in the same
  run.
- Flag recommendations that depend on facts not present in the tool outputs or
  the user prompt.
- Reproduce failures with a minimal prompt that preserves the drug, condition,
  and patient-specific constraints.
- For medication dosing questions, verify whether the relevant label,
  contraindication, impairment, or interaction tool was available in the
  selected tool set.

## Reporting an Issue

When opening an issue, include:

- exact prompt
- model names
- `init_rag_num`, `step_rag_num`, `call_agent`, and seed
- hardware and vLLM settings
- selected tools or tool-call trace
- final answer and the expected evidence source
