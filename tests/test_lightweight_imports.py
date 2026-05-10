from txagent import ToolRAGModel, TxAgent


def test_package_import_exposes_public_classes():
    import txagent

    assert txagent.__all__ == ["TxAgent", "ToolRAGModel"]


def test_txagent_constructor_does_not_load_models():
    agent = TxAgent("dummy-llm", "dummy-rag")

    assert agent.model is None
    assert agent.tooluniverse is None
    assert agent.rag_model.rag_model is None


def test_toolrag_constructor_does_not_load_embedding_model():
    rag_model = ToolRAGModel("dummy-rag")

    assert rag_model.rag_model is None
    assert rag_model.tool_desc_embedding is None
