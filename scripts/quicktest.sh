PYTHONPATH=. python evaluation/inference_baseline.py \
    --model-path google/gemma-2-9b-it \
    --model-id gemma-9b-it \
    --question-file data/spec_bench/split_categories/summarization.jsonl \
    --dtype bfloat16