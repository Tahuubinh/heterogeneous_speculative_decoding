import json
import os
from collections import defaultdict

# Path to the JSONL file
input_file = 'data/spec_bench/question.jsonl'
output_dir = 'data/spec_bench/split_categories'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Dictionary to hold categories and their respective questions
categories = defaultdict(list)

# Read the JSONL file and group questions by category
with open(input_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        category = data.get('category', 'unknown')
        categories[category].append(data)

# Write each category to a separate JSONL file
for category, questions in categories.items():
    output_file = os.path.join(output_dir, f"{category}.jsonl")
    with open(output_file, 'w') as out_file:
        for question in questions:
            out_file.write(json.dumps(question) + '\n')

print(f"Questions have been split by category into {output_dir}")