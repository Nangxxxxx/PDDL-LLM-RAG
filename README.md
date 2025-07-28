

# LLM Planning Pipelines

This repository provides tools to evaluate the capability of LLMs to generate PDDL files or plans from textual task descriptions.

---

## 🔧 Setup

Make sure you are using Python 3. Then install required dependencies (if any).

---

## 🚀 Run: LLM-as-Formalizer

To generate PDDL files from textual descriptions:

```bash
python3 source/llm-as-formalizer.py \
  --domain DOMAIN \
  --model MODEL \
  --data DATA \
  --index_start INDEX_START \
  --index_end INDEX_END
```

### Arguments

* `DOMAIN`: One of

  * `blocksworld`
  * `mystery_blocksworld`
  * `barman`
  * `logistics`
* `MODEL`: One of

  * `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`, `o1-preview`
  * `google/gemma-2-9b-it`, `google/gemma-2-27b-it`
  * `meta-llama/Meta-Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`, `meta-llama/Llama-3.1-405B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`
  * `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
  * `o3-mini`
* `DATA`: One of

  * `Heavily_Templated_BlocksWorld-100`
  * `Moderately_Templated_BlocksWorld-100`
  * `Natural_BlocksWorld-100`
  * `Heavily_Templated_Mystery_BlocksWorld-100`
  * `Heavily_Templated_Barman-100`
  * `Heavily_Templated_Logistics-100`
  * `Moderately_Templated_Logistics-100`
  * `Natural_Logistics-100`
* `INDEX_START`: Start index (inclusive, e.g., 1 means p01)
* `INDEX_END`: End index (exclusive, e.g., 112 means stop at p111)

### Output

Results will be saved to:

```
/outputs/llm-as-formalizer/DOMAIN/DATA/MODEL/
```

---

## 🧠 Run: LLM-as-Planner

To generate plans directly using LLMs:

```bash
cd source
python3 source/llm-as-planner.py \
  --domain DOMAIN \
  --model MODEL \
  --data DATA \
  --index_start INDEX_START \
  --index_end INDEX_END
```

### Output

Results will be saved to:

```
/outputs/llm-as-planner/DOMAIN/DATA/MODEL/
```

---

## 🧩 Run Solver

After generating PDDL using LLM-as-Formalizer, you can run a solver to generate plans:

```bash
python3 source/run_solver.py \
  --domain DOMAIN \
  --model MODEL \
  --data DATA \
  --index_start INDEX_START \
  --index_end INDEX_END \
  [--solver SOLVER]
```

* `SOLVER`: Optional, one of:

  * `lama-first`
  * `dual-bfws-ffparser` (default)

---

## 📊 Evaluation

To evaluate the output using [VAL](https://github.com/KCL-Planning/VAL), run:

```bash
python3 source/run_val.py \
  --domain DOMAIN \
  --model MODEL \
  --data DATA \
  --index_start INDEX_START \
  --index_end INDEX_END \
  --prediction_type PREDICTION_TYPE \
  [--csv_result]
```

### Arguments

* `PREDICTION_TYPE`: One of

  * `llm-as-formalizer`
  * `llm-as-planner`
* `--csv_result`: Optional flag to export results (plans + errors) to CSV.

### Output

CSV results will be saved to:

```
/outputs/PREDICTION_TYPE/DOMAIN/DATA/MODEL/
```

---

## 📁 Folder Structure

```
outputs/
├── llm-as-formalizer/
│   └── DOMAIN/DATA/MODEL/
├── llm-as-planner/
│   └── DOMAIN/DATA/MODEL/
└── ... (csv results if --csv_result is enabled)
```

