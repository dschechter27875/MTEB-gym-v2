# MTEB Gym

Label-free, LLM-judged evaluation of embedding models.

Give it a corpus, either an MTEB task or your own documents. It generates queries for the corpus, or takes yours. Every model retrieves for the same queries, an LLM judge compares the retrieved lists pairwise, and Bradley–Terry turns the comparisons into a ranking. No relevance labels are needed.

```text
corpus → queries → retrieval → pairwise LLM judging → ranking
```

Started in the [MTEB Gym discussion](https://github.com/embeddings-benchmark/mteb/discussions/3068).

## Installation

```bash
pip install "mteb-gym @ git+https://github.com/embeddings-benchmark/MTEB-gym-v2"
```

`[colbert]` adds late-interaction models.

## Quickstart

The judge and the generator are LLMs reached over an OpenAI-compatible API. You can run without one, with a hosted API, or with a model you serve yourself.

**Without an API key or GPU**, use the mock judge. It answers deterministically, so this only checks that everything is installed. It does not rank models and takes about a minute on a CPU.

```python
import mteb_gym as gym

result = gym.run(
    corpus="NanoNFCorpusRetrieval",
    models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
    judge=gym.MockLLM(),
    n_queries=20,
    output_folder="results/demo",
)
print(result.leaderboard)
```

**With an API key.** `gym.LLM(model)` uses OpenAI. Pass `base_url` for any other OpenAI-compatible provider.

```bash
export OPENAI_API_KEY=<your_api_key>
```

```python
result = gym.run(
    corpus="NFCorpus",
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],
    generator=gym.LLM("gpt-5.4-mini"),  # writes the queries
    judge=gym.LLM("gpt-5.4"),  # compares what the models retrieve
    n_queries=100,
    output_folder="results/nfcorpus",
)
print(result.leaderboard)
```

Or from the command line:

```bash
mteb-gym --corpus NFCorpus --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 --generator gpt-5.4-mini --judge gpt-5.4
```

**With an open model you serve yourself.** Start it with vLLM, `transformers serve` or Ollama and pass its URL. No key is needed, and one model can be both judge and generator.

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

```python
llm = gym.LLM("Qwen/Qwen3-4B-Instruct-2507", base_url="http://localhost:8000/v1")
result = gym.run(corpus="NFCorpus", models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5"], generator=llm, judge=llm)
```

## Usage

| `gym.run` argument | Takes |
|---|---|
| `corpus` | an MTEB retrieval task name, a directory of `.txt` / `.md` files, or a `.jsonl` with `id` and `text` |
| `models` | MTEB model ids; they run through mteb, so prompts, revisions and retrieval paths match an official run |
| `judge`, `generator` | LLM clients, see below; without a generator the judge writes the queries |
| `queries` | `"synthetic"` (default), `"original"` for an MTEB task's own queries, or your own as a `.jsonl` with `id` and `text`, a `.txt` with one per line, or a list of strings |
| `task_description` | one sentence on what counts as a good result, seen by generator and judge, e.g. `"Given a claim, find documents that refute it"`; defaults to the task's mteb prompt |
| `n_queries`, `top_k`, `seed` | 100 generated queries, 10 documents judged per query, seed 0 |
| `filter_queries` | LLM quality filter and deduplication, on by default |
| `output_folder`, `batch_size`, `workers` | where files go (`results`), encoding batch size, concurrent LLM calls |

**LLMs.** `gym.LLM(model)` uses OpenAI and reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`. `gym.LLM(model, base_url=..., api_key=...)` reaches any other OpenAI-compatible endpoint: vLLM, Ollama, Together, OpenRouter, Anthropic, Gemini. For an experiment, use a judge and a generator from different model families.

**Output.** Everything is written under `output_folder`:

```text
results/nfcorpus/
├── records/NFCorpus__gpt-5.4__gpt-5.4-mini__q100-s0-<hash>.json   # ratings, config, diagnostics
├── queries/       # generated queries with quality scores
├── predictions/   # mteb's retrieval output per model
└── verdicts/      # judge verdicts per model pair
```

A rerun of the same configuration reuses all of it, and adding a model judges only the new pairs. Judging makes two calls per query per model pair, so 100 queries and 10 models is 9,000 calls. `gym.Result.from_disk(path)` reads one run (`.leaderboard`, `.to_dataframe()`); `gym.load_results("results/")` reads every run under a directory.

**Agreement with MTEB.** For an MTEB task, the ranking can be compared with the official scores after the run. The labels never enter the pipeline. Running with `queries="original"` isolates the judge from query generation.

```python
result.agreement()  # one run
gym.load_results("results/").agreement()  # every run under a directory
```

Official scores come from the MTEB results repository through mteb's cache. `agreement(evaluate_missing=True)` runs mteb for models that have none. The output gives Spearman, Kendall, top-10 Spearman and AP correlation with bootstrap intervals, and says whether each score was official or self-run.

## How it works

1. Sample documents; the generator writes one query per sample at temperature 0.7. Short, malformed, low-quality and near-duplicate queries are dropped.
2. Every model retrieves through `mteb.evaluate`.
3. The judge compares two models' top-k lists per query, in both orders, at temperature 0. A split decision counts half.
4. Bradley–Terry over all pairwise outcomes gives the ranking; confidence intervals come from resampling queries.
5. Queries, predictions, verdicts, model revisions and configuration are written to disk and cached by identity, so a rerun repeats only what changed.

## Development

```bash
make install   # uv sync with dev tools
make test      # pytest, two end-to-end runs included
make lint      # ruff
```

Tests use the mock LLM: no key, no GPU. CI runs on Python 3.10 and 3.13.

```text
mteb_gym/
├── corpus.py     an MTEB task corpus or a local one
├── queries.py    query generation and filtering
├── task.py       corpus + queries as an mteb retrieval task
├── retrieval.py  mteb.evaluate per model; read its predictions
├── judge.py      pairwise judging, both orders
├── rank.py       Bradley–Terry with bootstrap CIs
├── results.py    Result, Results, load_results
├── agreement.py  agreement with official MTEB scores
├── llm.py        LLM and MockLLM
└── run.py        the pipeline
```

## Citation

Paper and citation information will be added with the MTEB Gym release.

MTEB Gym builds on MTEB:

```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```
