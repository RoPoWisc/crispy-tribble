Some notes on this repo:

This repo does not include the raw documents in the data directory that contain the original ad copies and reviews. Instead, we include some test documents to help you run the data pipeline. These are raw cached files from a supplemental test run we did with Gemini 2.5, provided so you can run the code with cached output without using a GCP Vertex key or any private offshoot of the original data.

For the data pipeline, you will need to run a local instance of LM Studio with the Gemma 3 12B model. You may run the model by other means or choose a different model entirely, but you will need to make appropriate updates to the code.

For the data synthesis pipeline, the default behavior is a cached run for convenience, but if you prefer to run the pipeline without the cached results, you will need to define key.json (your GCP service account with appropriate access).

merge_dpo_outputs.py is a script that helps you consolidate runs from run_trace_correction.py. This script exists to merge multiple synthesis runs because run_trace_correction.py tends to drop close to 87% of the synthesized samples to keep the DPO training set relatively high quality.

We provide a Colab notebook to fine-tune the Qwen 4B Thinking model with the DPO dataset. It showcases our fine-tuned model run, and we include the relevant dataset for inspection and for a subsequent validation run. We strongly recommend running on an A100; you may reduce the number of epochs compared to our run, as we trained for additional epochs to see whether we would encounter a double-descent phenomenon, which did not appear to be the case.

Our human evaluation set contains held-out samples from the synthesized corpus that we did not train on, as well as new synthetic data generated with the GPT-5.1 model. These additional samples were designed to mimic client-facing materials from RIAs, with intentional breakage similar to how we synthesized our main test corpus.

This code is provided for transparency and reproducibility, not for unrestricted free use. We strictly prohibit using this work or any derivative for commercial use. We welcome others to build on it for further exploration and research.


Quick start commands for the three entrypoints:
```
# end-to-end ingestion + sanitization
python3 run_full_pipeline.py \
    --limit 2 \
    --skip-pptx \
    --log-failures \
    --sanitize \
    --sanitize-output output/sanitized_text

# breaker/fixer synthesis with cache reuse
python3 run_breaker_fixer.py \
    --sanitized-root output/sanitized_text \
    --sanitized-limit 60 \
    --rules-limit 5 \
    --atomic-per-rule 8 \
    --near-miss-per-rule 2 \
    --compound-pairs 1 \
    --output output/synthetic/breaker_fixer.jsonl \
    --batch-fixer-cache output/cache/batch_fixer.jsonl \
    --key-file-path key.json

# procedural trace correction with cached Gemini responses
python3 run_trace_correction.py \
    --input output/synthetic/breaker_fixer.jsonl \
    --dpo-output output/experiments/output_procedural_dpo.jsonl \
    --key-file key.json
```
