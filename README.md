# When Do Graphs Help RAG?

An exploratory study built on the [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) dataset and evaluation code. It examines whether graph-derived context helps retrieval-augmented generation (RAG), whether graph relations matter, and how much graph context is useful.

This project is an experimental extension, not a complete reproduction of every method in the paper or benchmark.

## Study questions

- **Q1 - Graph structure:** Does graph-based retrieval help beyond flat retrieval, and what happens when graph content or topology is changed?
- **Q2 - Selection:** Do greedy or adaptive graph-context selectors improve the quality/cost trade-off?
- **Q3 - Relations:** Do graph links and their relation labels add value when graph conditions are matched?
- **Q4 - Context size:** What graph depth is a promising candidate for a smaller context?

## Status and main findings

The results below are transcribed from [`research/results/results.csv`](research/results/results.csv). They are exploratory, report-level summaries; Q1/Q2 do not include paired inference, and Q3/Q4 use paired analyses only where noted. Scores are on a 0-1 scale. For Q1/Q2, means pool the Novel and Medical corpus summaries weighted by their reported `n`. `†` marks a metric for which at least one corpus report is marked incomplete or unverified; interpret those means cautiously.

| Question | Work completed | Descriptive result |
|---|---|---|
| Q1 | Four retrieval conditions, two token budgets, 400 questions per corpus | RAG scored higher on answer correctness at both budgets. At 4,096 tokens, `graph_full`/`graph_rewired` had slightly higher evidence recall than RAG, but the aggregate difference is not paired or tested. |
| Q2 | Greedy/adaptive selection plus shared RAG and `graph_full` baselines | `graph_greedy` trailed RAG on correctness and evidence recall at both budgets. `graph_adaptive` at 4,096 tokens had the highest correctness mean, but its evidence recall was essentially tied with RAG. |
| Q3 | Six relation/topology conditions at 4,096 tokens; paired comparisons | Intact relations did not improve correctness over any matched graph condition. Evidence recall was higher than nodes-only and hidden predicates, but not distinguishable from permuted predicates or rewired topology in the unadjusted intervals. |
| Q4 | 11-condition preliminary screen on 80 questions, with paired bootstrap contrasts | No depth met the planned non-inferiority margins against **both** references on both evaluated metrics. The screen is exploratory, not a confirmation result. |

### Q1 - Graph structure

The six reported metrics are answer correctness, context relevance, evidence recall, coverage, faithfulness, and ROUGE-L. Q1 has report-level means for all six metrics; the results below combine Novel and Medical using each report's `n`. Coverage is reported over Contextual Summarize and Creative Generation, faithfulness over Creative Generation, and ROUGE-L over Fact Retrieval and Complex Reasoning, so their sample sizes are smaller than the all-type metrics.

| Condition | Budget | Answer correctness | Context relevance | Evidence recall | Coverage | Faithfulness | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rag` | 1,024 | 0.489† | 0.148 | 0.454 | 0.630† | 0.620† | 0.251† |
| `graph_content` | 1,024 | 0.437† | 0.031 | 0.416 | 0.591† | 0.596† | 0.214† |
| `graph_full` | 1,024 | 0.438 | 0.044 | 0.415 | 0.571 | 0.563 | 0.219 |
| `graph_rewired` | 1,024 | 0.450 | 0.026 | 0.426 | 0.578 | 0.558 | 0.215 |
| `rag` | 4,096 | 0.504 | 0.090 | 0.460 | 0.630 | 0.678 | 0.241 |
| `graph_content` | 4,096 | 0.429† | 0.018 | 0.419 | 0.564† | 0.576† | 0.219† |
| `graph_full` | 4,096 | 0.487† | 0.006 | 0.476 | 0.576† | 0.493† | 0.156† |
| `graph_rewired` | 4,096 | 0.487 | 0.004 | 0.477 | 0.592 | 0.520 | 0.155 |

The flat baseline led on answer correctness, context relevance, and ROUGE-L in the available aggregates. At 4,096 tokens, `graph_full` and `graph_rewired` had evidence-recall means about 0.016-0.017 above RAG, while correctness was about 0.017 below it. These are descriptive, unpaired differences. Some 4,096-token correctness, coverage, faithfulness, and ROUGE-L reports are marked incomplete or unverified, as flagged above. Q1 rewiring also does not preserve edge attributes in the implemented path; it is not a clean topology-only intervention.

### Q2 - Context selection

Q2 reuses the Q1 RAG and `graph_full` runs and adds greedy and adaptive selection. Metric applicability and aggregation are the same as in Q1.

| Condition | Budget | Answer correctness | Context relevance | Evidence recall | Coverage | Faithfulness | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rag` | 1,024 | 0.489† | 0.148 | 0.454 | 0.630† | 0.620† | 0.251† |
| `graph_full` | 1,024 | 0.438 | 0.044 | 0.415 | 0.571 | 0.563 | 0.219 |
| `graph_greedy` | 1,024 | 0.378 | 0.007 | 0.289 | 0.498 | 0.486 | 0.210 |
| `graph_adaptive` | 1,024 | 0.445 | 0.027 | 0.411 | 0.568 | 0.560 | 0.221 |
| `rag` | 4,096 | 0.504 | 0.090 | 0.460 | 0.630 | 0.678 | 0.241 |
| `graph_full` | 4,096 | 0.487† | 0.006 | 0.476 | 0.576† | 0.493† | 0.156† |
| `graph_greedy` | 4,096 | 0.467† | 0.003 | 0.367 | 0.575† | 0.560† | 0.153† |
| `graph_adaptive` | 4,096 | 0.508 | 0.003 | 0.460 | 0.580 | 0.572 | 0.161 |

`graph_greedy` is consistently below RAG on answer correctness and evidence recall. `graph_adaptive` at 4,096 tokens has a slightly higher correctness mean than RAG (0.508 vs. 0.504), but equal evidence recall (0.460) and lower context relevance; this does not establish an improvement. Q2 aggregate means have no paired confidence intervals.

### Q3 - Relation ablations

Q3 reports only answer correctness and evidence recall; the CSV has no Q3 values for context relevance, coverage, faithfulness, or ROUGE-L. Means below use the 397-question paired set except where a correctness judgment failed (`n` is 396 for `graph_intact` and 396 for `graph_nodes_only`).

| Condition | Answer correctness | Evidence recall |
|---|---:|---:|
| Intact triples | 0.502 | 0.482 |
| Nodes only | 0.524 | 0.430 |
| Hidden predicates | 0.496 | 0.426 |
| Permuted predicates | 0.515 | 0.487 |
| Rewired topology | 0.491 | 0.469 |
| RAG reference | 0.514 | 0.466 |

Paired differences (`graph_intact` minus comparison; 95% bootstrap interval) are shown below. The matched graph contrasts improve evidence recall over nodes-only and hidden predicates, but none shows a correctness advantage. The RAG comparison changes the retrieved content and is not a matched causal contrast.

| Comparison | Correctness difference (95% CI) | Evidence-recall difference (95% CI) |
|---|---:|---:|
| Hidden predicates | +0.005 [-0.021, +0.031] | +0.056 [+0.024, +0.087] |
| Nodes only | -0.023 [-0.051, +0.005] | +0.052 [+0.020, +0.086] |
| Permuted predicates | -0.014 [-0.040, +0.011] | -0.005 [-0.031, +0.020] |
| Rewired topology | +0.010 [-0.016, +0.037] | +0.013 [-0.014, +0.041] |
| RAG reference | -0.013 [-0.042, +0.017] | +0.016 [-0.021, +0.051] |

The intervals are unadjusted; the planned Holm correction was not applied. Treat the apparent evidence-recall advantages over nodes-only and hidden predicates as exploratory.

### Q4 - Context-depth screen

Q4 reports answer correctness, evidence recall, and mean context tokens; it does not report context relevance, coverage, faithfulness, or ROUGE-L. This preliminary screen used 10 questions from each of the eight corpus-by-question-type development strata (80 total), not the untouched confirmation split.

| Method | Depth | Mean context tokens | Answer correctness | Evidence recall |
|---|---:|---:|---:|---:|
| RAG reference | - | 2,552 | 0.526 | 0.515 |
| `graph_full` reference | - | 3,386 | 0.444 | 0.453 |
| `graph_full` | 1 | 3,042 | 0.484 | 0.403 |
| `graph_full` | 2 | 3,542 | 0.480 | 0.438 |
| `graph_full` | 3 | 3,703 | 0.500 | 0.458 |
| `graph_greedy` | 1 | 3,041 | 0.493 | 0.297 |
| `graph_greedy` | 3 | 3,704 | 0.480 | 0.288 |
| `graph_greedy` | 6 | 3,816 | 0.496 | 0.270 |
| `graph_adaptive` | 1 | 1,496 | 0.467 | 0.395 |
| `graph_adaptive` | 3 | 3,557 | 0.496 | 0.434 |
| `graph_adaptive` | 6 | 3,782 | 0.469 | 0.454 |

The greedy selector has substantially lower evidence recall at every screened depth. `graph_full` depth 3 and `graph_adaptive` depth 6 clear the planned non-inferiority margins against the `graph_full` reference on both measured metrics; neither clears the margins against RAG, so no setting satisfies the rule against **both** references. These paired bootstrap estimates are exploratory and do not replace the planned confirmation. The full 18-cell Q4 development generation exists; the full development evaluation and confirmation were not completed.

## Dataset and frozen sample

The source is GraphRAG-Bench's Novel and Medical corpora and question sets:

- **Novel:** questions are drawn from 20 books.
- **Medical:** one medical corpus.
- **Evaluation sample:** 400 questions: 200 per corpus and 50 per question type in each corpus.
- **Question types:** Fact Retrieval, Complex Reasoning, Contextual Summarize, and Creative Generation.
- **Sampling seed:** 42.
- **Frozen ID digest:** `6c6a38acaa73897913365bf7b8c3113e1ce84f438e66290d07aca329abf6c17a`, stored with the IDs in `research/configs/questions_400.json`.

The graph build manifest records 21 graphs, 6,051 chunks and 23,835 extracted triples. Graph data and derived corpora are local artifacts, not prerequisites to publish. Check the benchmark's license and download the source data from the official repository when reproducing.

## Graph construction

The local pipeline chunks each corpus source into 256-token windows with 32-token overlap. An LLM extracts entity-relation-entity triples per chunk; the pipeline validates, normalizes and deduplicates triples, then joins them into one graph per source. The build manifest reports 26,428 total nodes and 22,395 total edges across the 21 graphs.

The observed generation, extraction, and judging model was **Qwen2.5 3B** served locally through Ollama. The graph representation used in the experiments is an **undirected NetworkX graph** with relation text stored on edges. It therefore does not encode the direction of a relation in traversal. Graph extraction can omit, invent, merge, or normalize entities and relations; the recorded graph statistics are not a comprehensive graph-quality evaluation.

## Experimental design

All stages use the frozen questions, the same source corpus per question, `cl100k_base` token counting, and a requested context ceiling of 1,024 or 4,096 tokens unless otherwise noted. The observed local setup used temperature 0.

### Q1 - Graph structure and context

| Condition | Implemented context | Intended comparison |
|---|---|---|
| `rag` | Top-k TF-IDF text chunks, packed to the token budget | Flat text baseline |
| `graph_content` | Entity names from graph-retrieved edges, serialized without edges | Graph retrieval/content without explicit relations |
| `graph_full` | Query-ranked graph edges serialized with endpoints and relation labels, selected from a two-hop neighborhood | Full graph context |
| `graph_rewired` | Graph with edge swaps, using the same node set | Topology intervention |

In the implemented Q1 `graph_rewired` path, `networkx.double_edge_swap` is called without explicit transfer of edge attributes. Therefore, the saved implementation does **not** establish that only topology changed while all relation labels/content stayed matched. Interpret the Q1 rewiring comparison cautiously; Q3 has a separate, more controlled intervention.

### Q2 - Budgeted graph selection

Q2 uses `rag` and `graph_full` as shared baselines, plus:

- **`graph_greedy`:** iteratively selects candidate edges by new query-term coverage per token, with small degree and retrieval-score tie contributions. It does not use gold evidence during inference.
- **`graph_adaptive`:** propagates node relevance through the graph with hop decay and selects the context at the hop with the best query-term coverage / edge count rank, subject to the budget.

The Q2 implementation uses lexical TF-IDF and query-term overlap, not a learned evidence-coverage model.

### Q3 - Relation ablations

At 4,096 tokens, a base two-hop graph subgraph is selected and transformed into six conditions:

1. `graph_intact`: original endpoints and extracted relation labels.
2. `graph_nodes_only`: selected entity names without edges.
3. `graph_hidden_predicates`: same topology with relation text masked.
4. `graph_permuted_predicates`: selected relation labels shuffled across edges.
5. `graph_rewired_topology`: endpoint swaps that preserve the selected degree
   sequence and relation-label multiset where rewiring succeeds; the
   implementation reassigns predicate bundles across the rewired edges.
6. `rag_reference`: flat retrieval baseline; an external baseline, not a content-matched graph intervention.

The graph conditions target matched contrasts for links, relation meaning,
label-to-edge binding, and topology. The topology condition also reassigns
predicate bundles, so it changes endpoint-to-predicate bindings along with
connectivity and is not a topology-only intervention. The RAG comparison
changes content as well, so it is not by itself a causal test of graph relations.

### Q4 - Context depth

The planned grid is `graph_full`, `graph_greedy`, and `graph_adaptive` at depths 1-6 with a 4,096-token ceiling. It consists of 18 configurations on the 320-question development split. The frozen split reserves 80 questions for confirmation. The intended selection rule requires a candidate to be non-inferior to **both** Q1 RAG and Q1 `graph_full` on answer correctness (margin -0.03) and evidence recall (margin -0.05), using paired confidence intervals. The confirmation phase was not run, so no sufficient-depth claim is justified.

## Evaluation and metrics

Generation and retrieval metrics use the benchmark's official evaluator code. The local reports record Qwen2.5 3B via Ollama as the evaluator model. LLM-judged scores are not ground truth and can fail to return parseable judgments.

| Metric | Interpretation |
|---|---|
| ROUGE-L | Longest-common-subsequence overlap between generated and reference answers; the saved evaluator reports F-measure with stemming. |
| Answer correctness | LLM-judged correctness of the generated answer against the reference. |
| Coverage | LLM-judged fraction of factual statements extracted from the reference answer that the response covers. |
| Faithfulness | LLM-judged support of generated answer statements by the retrieved context. |
| Context relevance | LLM rating of how well retrieved context can answer the question; the evaluator averages two 0/1/2 ratings after normalization to 0-1. |
| Evidence recall | LLM-judged fraction of reference evidence items supported by the retrieved context. |
| Graph structure | Build manifest includes node/edge counts, average degree, average clustering, density, connected components, largest component, and isolated nodes. |
| Efficiency | Prediction records can store context/prompt tokens and latency. These were not consistently consolidated into the final summary. |

The official evaluator uses different generation metrics by question type: ROUGE-L and correctness for Fact Retrieval and Complex Reasoning; correctness and coverage for Contextual Summarize; correctness, coverage, and faithfulness for Creative Generation. Retrieval reports include context relevance and evidence recall for all four types. Graph helper functions for evidence-triple coverage and reachability exist, but those should not be described as completed results unless corresponding result artifacts are produced.

The analysis code includes paired-bootstrap and multiple-testing helpers. Q3 intervals reported above are unadjusted, Q1/Q2 archived evaluation files are aggregate-level, and Q4's planned confirmatory test is incomplete. Do not infer statistical significance from the aggregate means in this README.

## Runtime and compute notes

Approximate wall times reported during this project:

| Work | Observed time |
|---|---:|
| Graph construction | About 8 hours |
| Q1/Q2 generation | About 9 hours |
| Q3/Q4 generation | About 24 hours |
| Q1/Q2 evaluation | About 31 hours |
| Q3/Q4 evaluation | About 40 hours |

These are operator-reported estimates, not benchmarked runtimes from a clean repeat. Local Ollama was used, so no hosted API cost is reported.

## Limitations and future work

- Qwen2.5 3B is a small local model; its generation quality and its LLM-judge reliability limit conclusions. Several evaluation reports contain failed or malformed metric judgments.
- Q1/Q2 results are descriptive aggregate reports rather than paired per-question inference.
- Q3 paired analysis excludes three unrewirable IDs and lacks the planned multiplicity correction.
- Q3 graph contexts are brought to the same total token count with padding text;
  predicate slots are not individually length-matched, and the padding may
  affect model behavior. The rewired condition changes topology and
  predicate-to-edge assignments together, so its score difference cannot
  isolate topology alone.
- Q4 has only a preliminary 80-question screen; it does not include confirmation on the frozen holdout.
- The graph is undirected despite representing semantic relations; a directed, typed multigraph could preserve more information.
- LLM-based triple extraction can be noisy. Better entity resolution, relation normalization, extraction verification, and human audits are needed.
- Structural graph statistics alone do not measure semantic correctness. Future work should validate evidence-triple coverage, reachability, relation precision/recall, and redundancy against manually checked samples.
- TF-IDF and lexical query overlap are limited retrieval/selection strategies. Stronger embedding, reranking, and evidence-aware methods should be compared under matched budgets.
- The 400-question sample and one benchmark domain setup do not establish broad generalization.

## Reproducing the experiments

### 1. Obtain the project and benchmark

```bash
git clone https://github.com/<OWNER>/<PROJECT-REPOSITORY>.git
cd <PROJECT-REPOSITORY>
git clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git official/GraphRAG-Benchmark
```

Replace the project placeholder with the published repository URL. Check the benchmark license before redistributing any corpus data.

### 2. Create separate environments

Use Python 3.12, the version used by the recorded working environments. Run the research harness and official evaluator in separate environments to reduce dependency conflicts:

```bash
python3 -m venv .venvs/grb-min
source .venvs/grb-min/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python3 -m venv .venvs/grb-eval
source .venvs/grb-eval/bin/activate
python -m pip install --upgrade pip
pip install -r official/GraphRAG-Benchmark/requirements.txt
```

`requirements.txt` pins the research-harness dependencies and retains the requested Torch pin, although the current project source does not import Torch directly. The official evaluator has its own dependency file and should stay in the separate environment.

### 3. Prepare the local model and verify data

Install and start Ollama separately, then:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
source .venvs/grb-min/bin/activate
export OLLAMA_MODEL=qwen2.5:3b
export OLLAMA_BASE_URL=http://localhost:11434/v1
export LLM_PROVIDER=ollama
python scripts/check_llm.py --ping
python scripts/verify_sample.py
python -m unittest discover -s tests -v
```

The frozen sample is `research/configs/questions_400.json`. Do not regenerate or edit it when reproducing a run.

### 4. Build the graphs

Start with a no-model dry run; then build all 20 Novel graphs and the Medical graph. This makes local Ollama calls and can take many hours:

```bash
source .venvs/grb-min/bin/activate
python scripts/build_graphs.py --dry-run --split all
python scripts/build_graphs.py --split all --workers 2
python scripts/clean_graphs.py
python scripts/validate_graphs.py
```

`clean_graphs.py` removes metadata-derived edges **in place**. During a fresh
reproduction, it copies the newly built raw graphs to a backup only if that
backup directory does not exist. The old backup in this workspace was removed;
the remaining `research/graphs/` are cleaned graphs, so they cannot restore the
pre-cleaning graph state. Rebuild the graphs before cleaning if that raw state
is needed. Keep graph files and caches if you want to avoid repeating
extraction; they are machine-local and are not included as source code.

### 5. Generate and evaluate Q1/Q2

The supplied shell runners use the current 400-question sample and the 1,024/4,096-token budgets:

```bash
bash scripts/run_q1_full.sh
bash scripts/run_q2_full.sh
```

They expect the research environment and a running Ollama service. Q2 reuses the shared baseline predictions when available. Run the evaluator in the separate environment after predictions exist:

```bash
source .venvs/grb-eval/bin/activate
python scripts/run_q1q2_eval_batch.py \
  --num-samples 50 \
  --output-dir research/results/evaluation_q1_q2/reproduction \
  --splits novel medical \
  --methods rag graph_content graph_full graph_rewired graph_greedy graph_adaptive \
  --budgets 1024 4096
```

`--num-samples 50` means 50 samples **per question type per corpus**, or 400 questions per condition and budget.

### 6. Generate and evaluate Q3/Q4

The independent generator and evaluator preserve the original Q1/Q2 scripts. Run a dry structural preflight before model calls:

```bash
source .venvs/grb-min/bin/activate
python scripts/generate_q3_q4.py --stages q3 q4 --dry-run --limit-per-cell 1
python scripts/check_llm.py --ping
python scripts/generate_q3_q4.py --stages q3 q4
```

The default Q4 phase is development. Full Q4 development generation is 18 configurations x 320 questions. The Q4 section above describes the design and the reduced exploratory screen used in this project; the generator's `--help` output documents its phase and depth options. Do not use the 80-question preliminary screen as a substitute for untouched confirmation. Q4 confirmation and its evaluation workflow remain incomplete.

To evaluate all generated Q3 conditions and the full Q4 development grid, use the separate evaluator environment. This is a long-running job:

```bash
source .venvs/grb-eval/bin/activate
python scripts/run_q3q4_eval_batch.py --preflight-only
python scripts/run_q3q4_eval_batch.py --stages q3
python scripts/run_q3q4_eval_batch.py --stages q4
```

To regenerate the analysis from already completed reports:

```bash
python scripts/analyze_q3_results.py
python scripts/analyze_q4_preliminary.py
```

Analysis commands operate on saved report artifacts; they do not recreate omitted model outputs. The reduced Q4 preliminary screen is described separately in the protocol.

## Repository map

| Path | Contents |
|---|---|
| `Paper/` | Source paper PDF and project notes. |
| `official/GraphRAG-Benchmark/` | Cloned benchmark source, evaluator, requirements, and source datasets. Do not edit as part of this project. |
| `research/configs/` | Frozen question IDs, Q4 split, preliminary subset, and experiment configuration files. |
| `research/src/` | Project research harness: loading, chunking, extraction, graph construction, retrieval, generation, budgets, and experiment conditions. |
| `research/src/README.md` | Responsibilities and execution flow for each retained research module. |
| `research/graphs/` | Generated per-source graphs. Git-ignored. |
| `research/cache/` | Chunk, extraction, LLM, and embedding caches. Git-ignored and potentially expensive to rebuild. |
| `research/results/evaluation_q1_q2/` | Archived Q1/Q2 evaluator reports, grouped by run/sample. |
| `research/results/evaluation_q3_q4/` | Q3/Q4 evaluator inputs, reports, paired analysis, and preliminary analysis. |
| `research/results/evaluation_raw/` | Raw Q1/Q2 predictions and Q3/Q4 generation checkpoints. Git-ignored; preserve for auditing and resuming. |
| `research/results/results.csv` | Long-format table of available Q1-Q4 means, paired differences, confidence intervals, and report status. |
| `research/results/RESULTS_SUMMARY.txt` | Brief Spanish interpretation of the results and their limitations. |
| `research/results/` | Graph build/cleanup/validation reports, frozen-sample audit, and consolidated summaries. |
| `research/results/sampling_report.json` | Balance and ID-digest audit for the frozen 400-question sample. |
| `scripts/` | Command-line scripts for checks, graph building, experiment generation, evaluation, and analysis. |
| `tests/` | Unit and focused pipeline tests. |
| `requirements.txt` | Pinned dependencies for the project harness and data audit. |
| `.venvs/` | Local Python virtual environments. Git-ignored; recreate using the dependency files. |

## Important files

- `research/configs/questions_400.json` - frozen evaluation sample and ID digest.
- `research/results/graph_build.json` - graph build manifest and structural counts.
- `research/results/evaluation_q1_q2/q1q2_n50_final/` - archived Q1/Q2 evaluator reports; check the status column because some reports are incomplete.
- `research/results/evaluation_q3_q4/q3_paired_analysis.json` - Q3 paired analysis.
- `research/results/evaluation_q3_q4/q4_preliminary_analysis.json` - Q4 preliminary screen summary.
- `research/results/results.csv` - consolidated long-format results; regenerate with `python scripts/export_results_summary.py`.
- `research/results/RESULTS_SUMMARY.txt` - short interpretation and caveats.
- `research/results/evaluation_raw/` - preserved, non-aggregated model outputs and checkpoints.
- `scripts/generate_q3_q4.py` - independent Q3/Q4 generation entry point.
- `scripts/run_q3q4_eval_batch.py` - Q3/Q4 input validation and evaluation runner.
- `scripts/export_results_summary.py` - regenerates the consolidated CSV from saved evaluator reports.

## Data and artifact handling

Do not commit credentials, environment files, raw predictions, caches, local graphs, or unlicensed corpus data. The benchmark checkout is an external dependency to clone separately; do not bundle its code or datasets unless their licenses permit redistribution. Check the redistribution terms for the paper PDF in `Paper/` before publishing it. Raw predictions and checkpoints are kept under the ignored `research/results/evaluation_raw/`; compact evaluation reports, `results.csv`, and the short interpretation remain separate. The original graph backup, temporary logs, empty notebooks folder, and unused derived-data placeholder were removed after review.
