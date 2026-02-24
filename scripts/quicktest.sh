PYTHONPATH=. python evaluation/inference_sps.py \
    --model-path google/gemma-2-9b-it \
    --draft-model-path google/gemma-2-2b-it \
    --model-id gemma-test-quick \
    --bench-name spec_bench \
    --question-begin 0 \
    --question-end 10 \
    --answer-file output/quick_test.jsonl \
    --dtype bfloat16 \
    --answer-file data/spec_bench/split_categories/summarization.jsonl