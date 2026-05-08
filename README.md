# PubMed Biomarker Search + NER

## What This App Is For

This project helps you do one workflow:

1. Search PubMed with a gene or disease keyword  
2. Retrieve paper metadata + abstract  
3. Run BioBERT NER on each abstract  
4. Get structured outputs for downstream analysis (table + CSV)

This is useful when you want to quickly build a candidate evidence set for biomarker analysis.

## What You Input

- A keyword query (examples: `BRCA1 breast cancer`, `EGFR lung adenocarcinoma`)
- Number of papers to retrieve
- Year range (`Year from`, `Year to`)
- Journal filter (optional, exact journal name)
- Model path (default: `outputs/best_model`)

## What You Get

The app returns two tables:

1. **Papers table**
   - `pmid`, `title`, `journal`, `year`, `abstract`
   - `entity_count` (how many entities detected)
   - `entity_types` (type summary, e.g. `Disease:3, Chemical:1`)

2. **Entities table**
   - `pmid`
   - `entity_type`
   - `entity_text`
   - `token_start`, `token_end`

Both tables can be downloaded as CSV.

## How to Interpret Labels (`entity_type` vs `label_id`)

- `entity_type`: readable category for analysis (use this in downstream work)
- `label_id`: internal numeric class index from model output

`label_id` does not carry biological meaning by itself.  
It must be interpreted through model mapping (`id2label`).

BIO convention:
- `B-XXX`: beginning of entity
- `I-XXX`: continuation of entity
- `O`: non-entity token

For detailed explanation, see [LABEL_GUIDE.md](LABEL_GUIDE.md).

## Scope and Limits

- This app does **retrieval + NER extraction**, not ranking biomarkers by biological validity.
- It uses abstract text only (not full-text articles).
- Entity quality depends on your trained model in `outputs/best_model`.

## Quick Start

From repo root:

```bash
cd /path/to/biobert_biomarker_ner
```

Create a clean environment (recommended):

```bash
conda create -n biomarker310 python=3.10 -y
conda activate biomarker310
pip install -r requirements.txt
```

Run app:

```bash
python -m streamlit run demo/app.py --server.port 8502
```

Open:

`http://localhost:8502`

## How To Use the App (Step-by-Step)

1. In **Model checkpoint path**, keep `outputs/best_model` (or set your own model folder).  
2. In **Search keyword**, enter gene/disease query.  
3. Set **Number of papers**.  
4. Click **Search and Extract**.  
5. (Optional) Apply entity filters:
   - choose entity types
   - set minimum entities per paper  
6. Review the two tables and download CSV for analysis.

## Recommended Query Patterns

- `GENE disease` (e.g., `TP53 ovarian cancer`)
- `disease biomarker`
- Add context words: `prognosis`, `diagnosis`, `therapy response`

## Files and Roles

- `demo/app.py`: Streamlit user interface
- `src/ingestion/pubmed_client.py`: PubMed search + fetch
- `src/extraction/bc5cdr_pipeline.py`: Task A wrapper for gene-disease evidence
- `src/extraction/jnlpba_pipeline.py`: Task B wrapper for biomedical entity discovery
- `src/retrieval/structured_query.py`: shared query-time pipeline (`query -> papers -> ner -> tables`)
- `src/extraction/ner_infer.py`: model inference and entity aggregation
- `src/extraction/train_ner.py`: NER training pipeline
- `pipelines/run_train.py`, `pipelines/run_eval.py`: runnable training/evaluation entrypoints
- `pipelines/run_extract_bc5cdr.py`, `pipelines/run_extract_jnlpba.py`: task-specific entrypoints
- `doc/system_architecture_diagram.md`: quick architecture diagram

## Troubleshooting

### 1) `ModuleNotFoundError: No module named 'src'`
Run from project root with:

```bash
python -m streamlit run demo/app.py
```

### 2) Streamlit page says disconnected / cannot connect
- Ensure the terminal process is still running.
- Restart on a different port:

```bash
python -m streamlit run demo/app.py --server.port 8503
```

### 3) `seqeval` install fails in Python 3.13
Use Python 3.10/3.11 environment as shown above.

### 4) No papers or no entities returned
- Try broader query terms.
- Increase paper count.
- Check that model path points to a valid trained checkpoint.

## Optional: Retrain the NER Model

```bash
python -m pipelines.run_train
```

Artifacts:
- BC5CDR best model: `outputs/best_model/`
- JNLPBA best model: `outputs/best_model_jnlpba/`
- BC5CDR reports: `outputs/reports/bc5cdr/`
- JNLPBA reports: `outputs/reports/jnlpba/`

Config files:
- `configs/bc5cdr.json`
- `configs/jnlpba.json`

## Task Entrypoints

Task A, gene-disease evidence:

```bash
python -m pipelines.run_extract_bc5cdr --query "BRCA1 breast cancer"
```

Local smoke check for Task A:

```bash
python -m pipelines.run_extract_bc5cdr --smoke
```

Task B, biomedical entity discovery workflow:

```bash
python -m pipelines.run_extract_jnlpba --query "IL-2 gene expression"
```

Local smoke check for Task B:

```bash
python -m pipelines.run_extract_jnlpba --smoke
```

## Baseline Export (Task A)

Export a fixed BC5CDR baseline snapshot for regression checks:

```bash
python -m pipelines.run_export_bc5cdr_baseline
```

Generated files:
- `outputs/reports/baseline_bc5cdr_papers.csv`
- `outputs/reports/baseline_bc5cdr_entities.csv`

## Baseline Export (Task B)

Export a fixed JNLPBA baseline snapshot for regression checks:

```bash
python -m pipelines.run_export_jnlpba_baseline --smoke
```

Generated files:
- `outputs/reports/baseline_jnlpba_papers.csv`
- `outputs/reports/baseline_jnlpba_entities.csv`

## Normalization Mappings

Build normalization mapping CSVs from downloaded raw sources (`HGNC`, `MeSH`, `ChEBI`):

```bash
python -m pipelines.build_normalization_mappings
```

Default outputs:
- `data/processed/normalization/gene_aliases.csv`
- `data/processed/normalization/disease_aliases.csv`
- `data/processed/normalization/chemical_aliases.csv`

Disease filtering (default is already enabled):
- Keeps only MeSH descriptor tree prefixes `C` and `F03` for disease aliases.

Override disease tree prefixes:

```bash
python -m pipelines.build_normalization_mappings --disease_tree_prefixes C
```

Override output directory:

```bash
python -m pipelines.build_normalization_mappings --outdir data/processed/normalization
```

## SQLite KB

Initialize SQLite schema:

```bash
python -m pipelines.run_init_sqlite
```

Ingest Task pipeline outputs into SQLite:

```bash
python -m pipelines.run_ingest_to_sqlite --task bc5cdr --smoke
python -m pipelines.run_ingest_to_sqlite --task jnlpba --smoke
```

Query SQLite KB:

```bash
python -m pipelines.run_query_sqlite --mode pmid --pmid SMOKE001
python -m pipelines.run_query_sqlite --mode normalized_id --normalized_id HGNC:1100
python -m pipelines.run_query_sqlite --mode type_keyword --entity_type Gene --keyword brca
```

## Module Smoke Checks (Old-school)

Run each core module directly:

```bash
python -m src.ingestion.pubmed_client --query "BRCA1 breast cancer" --retmax 3
python -m src.extraction.data --dataset bc5cdr --max_length 128
python -m src.extraction.train_ner --dry_run
python -m src.retrieval.structured_query --query "BRCA1 breast cancer" --retmax 3
python -m src.extraction.bc5cdr_pipeline --smoke
python -m src.extraction.jnlpba_pipeline --query "IL-2 gene expression"
```

## Unit Tests

Run minimal unit tests:

```bash
python -m pytest -q tests/unit
```

Current unit tests focus on:

- PubMed XML parsing (mocked)
- Retrieval pipeline assembly (mocked)
- BIO span-to-entity merge behavior
- Unified task output schema regression (BC5CDR + JNLPBA column contract checks)

Schema contract source:
- `src/contracts/task_output_schemas.py` (shared definitions)
- `src/contracts/registry.py` (versioned registry, e.g. `bc5cdr:v1`, `jnlpba:v1`)
