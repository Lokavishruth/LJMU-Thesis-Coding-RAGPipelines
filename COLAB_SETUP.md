# Running the notebooks in Google Colab

All 90 notebooks use **absolute Google Drive paths**, so where you put the
`.ipynb` files (or how deeply they're nested) does not matter — they always
resolve data and credentials through `BASE_DIR`:

```python
BASE_DIR = '/content/drive/MyDrive/HotPotQA-Coding-Trials'
```

## 1. Required layout on Google Drive

Create this folder in **My Drive** and put two files in it:

```
MyDrive/HotPotQA-Coding-Trials/
├── api_keys.csv      ← your Groq API keys  (column header: API_KEY)
├── .env              ← your vector-DB credentials (see below)
└── results/          ← created automatically by the notebooks
```

> The notebook files themselves do **not** need to be on Drive. You can open
> them straight from GitHub/upload, or mirror the `notebooks/` tree onto Drive —
> either works, because nothing is read relative to the notebook's location.

## 2. The `.env` file

Create `MyDrive/HotPotQA-Coding-Trials/.env` with these keys:

```dotenv
QDRANT_KEY=<your-current-qdrant-cloud-api-key>
CLUSTER_EP=<your-qdrant-cluster-url>      # e.g. https://xxxx.eu-west-2-0.aws.cloud.qdrant.io:6333
PINECONE_KEY=<your-pinecone-api-key>
PINECONE_384=<host of your 384-dim Pinecone index>   # used by all-MiniLM-L6-v2 (DataBaseComparison/Pinecone)
PINECONE_768=<host of your 768-dim Pinecone index>   # used by e5-base-v2 (BestEmbeddingComparison/Pinecone)
GROQ_API_KEY=<optional; Groq keys actually come from api_keys.csv>
```

Each notebook now mounts Drive and runs, right after the mount cell:

```python
from dotenv import load_dotenv
load_dotenv(f'{BASE_DIR}/.env')
```

- **Qdrant** notebooks read `os.getenv("CLUSTER_EP")` and `os.getenv("QDRANT_KEY")`.
- **Pinecone** notebooks connect to a **pre-created** serverless index by host,
  chosen by embedding dimension:
  `index = pc.Index(host=os.getenv(f"PINECONE_{PINECONE_DIM}"))`. So a 384-dim
  model pulls `PINECONE_384` and a 768-dim model pulls `PINECONE_768`. Create the
  two indexes in the Pinecone console first and paste their **Host** values into
  `.env`. (The notebooks no longer create indexes themselves.)
- **Groq** keys are loaded from `api_keys.csv` (the thread-safe key carousel).

There is a single Qdrant credential pair — point `CLUSTER_EP`/`QDRANT_KEY`
at whichever cluster is currently live. Qdrant collections are created and
deleted on every run, so no cluster holds persistent data.

## 3. What gets written where

Per-run JSON/log files go to `RESULTS_DIR`; each group also appends a row to a
single **master metrics CSV**:

| Notebook group | `RESULTS_DIR` on Drive | Master metrics CSV |
|---|---|---|
| `DataBaseComparison/<DB>/` | `results/<db>` (faiss, lancedb, chromadb, qdrant, pinecone) | `results/DataBaseComparison_results.csv` |
| `BestEmbeddingComparison/<DB>/` | `results/e5_trials/<db>` | `results/experiment_metrics_e5.csv` |
| `EmbeddingModelsComparison/<model>/` | `results/embedding_models/<model>` | `results/EmbeddingModelsComparison_results.csv` |

Each group writes to its own CSV (no more cross-folder collisions), and all 5
model subfolders in `EmbeddingModelsComparison` append to the one shared file,
distinguished by the pipeline-label column.

## 4. Run order per pipeline folder

Run the cells top-to-bottom. The first cell `!pip install …` installs everything
(now including `python-dotenv`). HotPotQA is pulled at runtime via
`load_dataset("hotpotqa/hotpot_qa", "distractor", …)` — no data upload needed.

For GraphRAG notebooks (`05`, `06`) the spaCy model is downloaded in-notebook:
`python -m spacy download en_core_web_sm`.

## ⚠️ Security note

The old (pre-restructure) notebooks committed a hardcoded Qdrant key, which is
still in git history (commit `First Push`). Treat that key as compromised and
rotate it if it isn't already the expired one. Going forward, `.env` and
`api_keys.csv` are gitignored, so credentials stay out of the repo.
