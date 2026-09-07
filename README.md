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

A dry run first. The mock judge answers deterministically, so this only checks that everything is installed. It does not rank models and takes about a minute on a CPU, with no API key and no GPU.

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

A real run. The judge and the generator are LLMs; `gym.LLM(model)` uses OpenAI, and the next section shows every other server it can talk to.

```bash
export OPENAI_API_KEY=<your_api_key>
```

```python
import mteb_gym as gym

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
mteb-gym --corpus NFCorpus \
    --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 intfloat/e5-base-v2 \
    --generator gpt-5.4-mini --judge gpt-5.4
```

## LLMs

`gym.LLM(model, base_url=None, api_key=None)` talks to any OpenAI-compatible server. Without `base_url` it uses OpenAI and reads `OPENAI_API_KEY`. For a hosted provider such as OpenRouter, Together, Anthropic or Gemini, pass its URL and key:

```python
gym.LLM("<model>", base_url="<provider url>", api_key="<key>")
```

To serve an open model yourself, with no key, start it with one of these.

vLLM:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

SGLang:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

transformers:

```bash
pip install "transformers[serving]"
transformers serve --force-model Qwen/Qwen3-4B-Instruct-2507 --port 8000
```

Ollama, which uses its own model names and port:

```bash
ollama pull qwen3:4b
```

```python
llm = gym.LLM("qwen3:4b", base_url="http://localhost:11434/v1")
```

Then point `gym.LLM` at the server. One model can be both judge and generator.

```python
import mteb_gym as gym

llm = gym.LLM("Qwen/Qwen3-4B-Instruct-2507", base_url="http://localhost:8000/v1")

result = gym.run(
    corpus="NFCorpus",
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],
    generator=llm,
    judge=llm,
    n_queries=100,
    output_folder="results/nfcorpus",
)
print(result.leaderboard)
```

For an experiment, use a judge and a generator from different model families.

## Usage

```python
gym.run(
    corpus,                  # MTEB task name, a folder of .txt/.md files, or a .jsonl with id and text
    models,                  # MTEB model ids, run through mteb itself
    judge,                   # gym.LLM(...)
    generator=None,          # gym.LLM(...); None: the judge writes the queries
    queries="synthetic",     # "original": the task's own queries; or a .jsonl, .txt or list of yours
    task_description=None,   # what counts as a good result, one sentence; None: the task's mteb prompt
    n_queries=100,           # generated queries
    top_k=10,                # documents judged per query
    seed=0,
    filter_queries=True,     # LLM quality filter and deduplication
    output_folder="results",
    batch_size=32,           # encoding
    workers=8,               # concurrent LLM calls
)
```

Everything is written under `output_folder`. The record holds the ratings, the configuration and the diagnostics:

```text
results/nfcorpus/
├── records/NFCorpus__gpt-5.4__gpt-5.4-mini__q100-s0-<hash>.json
├── queries/       # generated queries with quality scores
├── predictions/   # mteb's retrieval output per model
└── verdicts/      # judge verdicts per model pair
```

- **Reruns.** The same configuration reuses all of it; adding a model judges only the new pairs.
- **Cost.** Two judge calls per query per model pair: 100 queries and 10 models is 9,000 calls.
- **Reading back.** `gym.Result.from_disk(path)` for one run (`.leaderboard`, `.to_dataframe()`); `gym.load_results("results/")` for every run under a directory.

**Agreement with MTEB.** For an MTEB task, compare the ranking with the official scores after the run. The labels never enter the pipeline; `queries="original"` isolates the judge from query generation.

```python
result.agreement()  # one run
gym.load_results("results/").agreement()  # every run under a directory
```

- Official scores come from the MTEB results repository through mteb's cache; `agreement(evaluate_missing=True)` runs mteb for models that have none.
- Reported: Spearman, Kendall, top-10 Spearman and AP correlation with bootstrap intervals, and whether each score was official or self-run.

## How it works

1. **Generate queries**  
   Sample documents; the generator writes one query per sample at temperature 0.7.

2. **Filter queries**  
   Drop short, malformed, low-quality and near-duplicate queries.

3. **Retrieve**  
   Every model retrieves through `mteb.evaluate`.

4. **Judge pairwise**  
   The judge compares two models' top-k lists per query, in both orders, at temperature 0. A split decision counts half.

5. **Rank models**  
   Bradley–Terry over all pairwise outcomes; confidence intervals from resampling queries.

6. **Record the run**  
   Queries, predictions, verdicts, model revisions and configuration go to disk, cached by identity, so a rerun repeats only what changed.

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
