import pandas as pd

from src.normalization.rule_based import normalize_entities_df


def test_normalize_entities_df_alias_and_fallback():
    entities_df = pd.DataFrame(
        [
            {"pmid": "1", "entity_type": "Gene", "entity_text": "BRCA1", "token_start": 0, "token_end": 0},
            {"pmid": "1", "entity_type": "Disease", "entity_text": "Rare syndrome", "token_start": 2, "token_end": 3},
        ]
    )

    out = normalize_entities_df(entities_df)

    # Known alias: BRCA1 is resolved to deterministic canonical ID.
    assert out.iloc[0]["normalized_text"] == "BRCA1"
    assert out.iloc[0]["normalized_id"] == "HGNC:1100"
    assert out.iloc[0]["normalized_source"] == "rule_alias_v1"
    assert out.iloc[0]["normalized_score"] == 1.0

    # Unknown surface form: preserve cleaned text and mark unresolved.
    assert out.iloc[1]["normalized_text"] == "rare syndrome"
    assert out.iloc[1]["normalized_id"] == "UNRESOLVED"
    assert out.iloc[1]["normalized_source"] == "rule_fallback_v1"
    assert out.iloc[1]["normalized_score"] == 0.5

