# Usage guide

A step-by-step walkthrough of the Retrieval-Augmented Literature Engine.

## 1. Start

```bash
cd rag_pipeline
source .venv/bin/activate
python build_index.py     # index the bundled sample corpus (first run only)
python app.py
```

Open **http://127.0.0.1:8000** in a browser. The badge in the top right shows the active corpus
(paper and chunk counts), the language model, and the embedding model in use.

## 2. The three-step flow

The interface is a **locked wizard**: a step does not open until the previous one is complete. The
steps follow the natural order of the SLR process.

### Step 1 · Indexing

Builds the paper corpus to work on. Three ways:

- **From the existing pool:** enter how many papers to index and press *Index*. (Uses the bundled
  sample corpus, or an external baseline corpus if one is mounted.)
- **Upload PDFs:** select one or more PDFs; the system extracts the text and indexes it.
- **Fetch from a database:** enter an arXiv search query (e.g. `hybrid retrieval augmented
  generation`) and a result count; the system fetches titles and abstracts and indexes them. With
  the *full text* switch on, PDFs are downloaded as well (slower, richer).

When indexing finishes, the paper count, chunk count, and duration are reported. The **active
corpus bar** at the top always shows which corpus you are working on. Every later step runs on that
corpus.

### Step 2 · PRISMA

The screening stage. Enter inclusion (IC) and exclusion (EC) criteria as keywords or sentences
(e.g. `RAG, reranking` or `Include only studies in the healthcare domain`). Select the screening
pool from the corpus (by ticking/filtering) and press *Generate PRISMA flow*.

The system applies two-stage RAG screening:

1. **Screening (title + abstract):** for each paper the chunks most relevant to the IC/EC are
   retrieved; a decision is made with its rationale.
2. **Eligibility (expanded context):** papers that pass screening are re-assessed with broader
   context.

The output is a **PRISMA 2020 flow diagram** (identification → screening → eligibility → included)
and a list of exclusion reasons. The included set is passed to the next step.

### Step 3 · Report

The synthesis and reporting stage. Enter one or more **research questions** and press *Generate
report*. Working only from the papers included by PRISMA, the system:

- Extracts the relevant fields from each paper **with a source quote** (no fabrication; unknown
  fields are left empty).
- Collects the relevant evidence across the corpus for each research question and produces a
  cited synthesis (`[P003]`, `[P005]`, ...).

The `[P###]` citations in the text are **clickable**: a reference card with author/year/title/DOI
opens. Export options are below the report.

## 3. Export and integrity

The **CSV / JSON / PDF** buttons below the report export the full output. Each output carries a
**SHA-256 checksum**. The `/api/verify` endpoint recomputes it to check whether a downloaded report
has been altered.

## 4. Notes

- PRISMA and report generation make real language-model calls; input/output token counts and
  duration are shown for each response.
- Indexing a new corpus replaces the previous one; the active set is always shown in the corpus bar.
- If `OPENROUTER_API_KEY` is not set, the language-model stages fail with a clear error (indexing
  and raw retrieval are unaffected and run offline with a local embedding model).
- The system listens on 127.0.0.1 and is not configured for remote access. It is designed for a
  single-user / laptop scenario.
