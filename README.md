

# LLM Planning Pipelines

This repo provides multiple pipelines that enhance PDDL generation through documentation.

---


## 🚀 Running Example: LLM-as-Formalizer

To generate PDDL files from textual descriptions:

```bash
python source/llm-as-formalizer.py \
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

  * `deepseek/deepseek-r1-distill-llama-70b`, `deepseek/deepseek-r1-distill-qwen-14b`
  * `meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct`
  * `QwQ-32B`, `Qwen3-8B`
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
python source/run_val.py \
  --domain DOMAIN \
  --model MODEL \
  --data DATA \
  --index_start INDEX_START \
  --index_end INDEX_END \
  --prediction_type PREDICTION_TYPE \
  [--csv_result]
```

