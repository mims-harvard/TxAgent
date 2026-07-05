"""Public package exports for TxAgent.

Imports are resolved lazily so lightweight modules can be used without
importing optional UI/runtime dependencies.
"""

__all__ = ["TxAgent", "ToolRAGModel"]


def __getattr__(name):
    if name == "TxAgent":
        from .txagent import TxAgent

        return TxAgent
    if name == "ToolRAGModel":
        from .toolrag import ToolRAGModel

        return ToolRAGModel
    raise AttributeError(f"module 'txagent' has no attribute {name!r}")
