import argparse
import time
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

def str_to_torch_dtype(dtype_str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    return mapping.get(dtype_str, torch.bfloat16)

def run_native_sps():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Target model (e.g., Gemma-9B)")
    parser.add_argument("--drafter-path", type=str, required=True, help="Draft model (e.g., Gemma-2B)")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # Load Tokenizer (Shared between 9B and 2B)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Target Model
    print(f"Loading Target Model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Load Draft Model
    print(f"Loading Draft Model: {args.drafter_path}")
    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Load questions
    questions = []
    with open(args.question_file, "r") as f:
        for line in f:
            questions.append(json.loads(line))

    results = []
    print("Starting Inference...")
    for q in tqdm(questions):
        prompt = q["turns"][0] if "turns" in q else q["text"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Start timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Native HF Assisted Generation
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                assistant_model=drafter,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # Greedy decoding for benchmarking
                use_cache=True
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        new_tokens = len(output_ids[0]) - len(inputs.input_ids[0])
        tps = new_tokens / total_time if total_time > 0 else 0

        results.append({
            "question_id": q.get("question_id", q.get("id")),
            "total_time": total_time,
            "new_tokens": new_tokens,
            "tps": tps,
            "answer": tokenizer.decode(output_ids[0], skip_special_tokens=True)
        })

    # Save results
    output_path = f"data/model_answer/{args.model_id}_native_sps.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"Inference complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_native_sps()