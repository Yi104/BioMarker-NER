import pytest

from src.contracts.registry import get_schema
from src.extraction.bc5cdr_pipeline import run_bc5cdr_pipeline
from src.extraction.jnlpba_pipeline import run_jnlpba_pipeline


@pytest.mark.parametrize(
    "schema_key,runner,query",
    [
        ("bc5cdr:v2", run_bc5cdr_pipeline, "BRCA1 breast cancer"),
        ("jnlpba:v2", run_jnlpba_pipeline, "IL-2 gene expression"),
    ],
)
def test_task_output_schema_is_stable(schema_key, runner, query):
    papers_df, entities_df = runner(query=query, smoke=True)
    schema = get_schema(schema_key)

    assert list(papers_df.columns) == schema["papers_columns"]
    assert list(entities_df.columns) == schema["entities_columns"]
