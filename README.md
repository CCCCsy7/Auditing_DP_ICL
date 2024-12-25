# AUDITING DIFFERENTIAL PRIVACY IN-CONTEXT LEARNING

This repository contains the code for our paper AUDITING DIFFERENTIAL PRIVACY IN-CONTEXT LEARNING, as part of a class project for CSCI 699 with Sai Praneeth Karimireddy.

## Installation

Use python 3.9. Note that installation order is important.
Install required packages with pip install -r requirements.txt

## Dataset

### Samsum Dataset

- **HF Repo**: (https://huggingface.co/datasets/Samsung/samsum)

## Run

### Dataset Pre-process

We randomly chose 1000 samples from the original training set as our database while 100 test sampels from the original test set to test our results.

```bash
python ./preprocess.py --dataset samsum
```

### Generate Queries

Generate all the queries used in Phase I-III.

```bash
python generate.py --exampler [exampler_size] --ensemble [ensemble_size] --llm [llm_name]
```

In our setup, we use `exampler_size = 0` as baselines and 3-shot (`exampler_size = 3`) for others. We used `ensemble_size = 10` for saving time and cost of querying the API. We could choose OpenAI API or Anthropic API based on `llm_name`.

### Get Embeddings

Get embeddings of all the outputs for Phase IV.

```bash
python get_embeddings.py --input_file [e.g. gpt_0shot_10ensemble_0canary.jsonl]
```

The `input_file` is the name of files under `./data/original_output`.


### Private Aggregation

Phase IV: Private Aggregation.

```bash
python ./aggregate_and_output.py --candidates_json gpt_0shot_10ensemble_0canary.jsonl --target_embeddings [gpt_3shot_10ensemble_25canary_embeddings.npy] --ensemble [10] --private [1] --eps [8]
```

Note that `--candidates_json` must be `pt_0shot_10ensemble_0canary.jsonl` since we used 0-shot outputs as our candidates to choose from during the embedding2text phase.

### Auditing

Audit the effectiveness of DP-ICL mechanisms by determining whether inserted canaries can be inferred from privatized results.

```bash
python ./prompt/prompt.py --eps [epsilon_value] --canary_size [canary_size]
```

Mechanisms of epsilon 1.0, 3.0, and 8.0 were audited, and canary sizes of 10, 25, 50, 100, 150, and 200 were used for each epsilon. Files for auditing procedure require phases I-IV to be run first.