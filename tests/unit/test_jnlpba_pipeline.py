import pandas as pd

from src.extraction import jnlpba_pipeline as jp


def test_run_jnlpba_pipeline_forwards_arguments(monkeypatch):
    sentinel_papers = pd.DataFrame([{"pmid": "1"}])
    sentinel_entities = pd.DataFrame([{"pmid": "1", "entity_type": "Protein"}])

    def fake_run_search_ner_pipeline(**kwargs):
        assert kwargs["query"] == "IL-2 gene expression"
        assert kwargs["model_path"] == "outputs/best_model_jnlpba"
        assert kwargs["retmax"] == 5
        assert kwargs["max_length"] == 128
        assert kwargs["year_from"] == 2010
        assert kwargs["year_to"] == 2020
        assert kwargs["journal"] == "Nature"
        return sentinel_papers, sentinel_entities

    monkeypatch.setattr(jp, "run_search_ner_pipeline", fake_run_search_ner_pipeline)
    papers_df, entities_df = jp.run_jnlpba_pipeline(
        query="IL-2 gene expression",
        model_path="outputs/best_model_jnlpba",
        retmax=5,
        max_length=128,
        year_from=2010,
        year_to=2020,
        journal="Nature",
    )

    assert papers_df.equals(sentinel_papers)
    assert entities_df.equals(sentinel_entities)
