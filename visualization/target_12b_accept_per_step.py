import matplotlib.pyplot as plt
import os
import json
import math
from pathlib import Path

def load_avg_accept_tokens(metrics_path):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    metrics = payload.get('metrics', {})
    value = metrics.get('avg_accept_tokens_per_step')
    if value is None:
        raise KeyError(f"avg_accept_tokens_per_step not found in {metrics_path}")
    return float(value)


def build_accept_lengths(results_root, target_model, task_map, drafter_names):
    accept_lengths = {}
    for drafter in drafter_names:
        backend_dir = 'mamba-replay' if 'mamba' in drafter.lower() else 'transformer'
        drafter_dir = Path(results_root) / target_model / drafter / 'adaptive'

        values = []
        for task_label, task_dir in task_map.items():
            spec_metrics = drafter_dir / task_dir / backend_dir / 'metrics.json'
            if not spec_metrics.exists():
                print(f"Warning: missing file {spec_metrics}")
                values.append(math.nan)
                continue

            avg_tokens = load_avg_accept_tokens(spec_metrics)
            values.append(avg_tokens)

        accept_lengths[drafter] = values

    return accept_lengths


def plot_avg_accept_tokens(tasks, drafter_names, accept_lengths):
    """
    Create a line plot for average accepted tokens per step and save it to the 'figures' folder.

    Parameters:
        tasks (list of str): List of task names.
        drafter_names (list of str): List of drafter names.
        accept_lengths (dict): Dictionary where keys are drafter names and values are lists of avg accept tokens for each task.
    """
    # Ensure the 'figures' folder exists
    os.makedirs('figures', exist_ok=True)

    # Create the line plot
    plt.figure(figsize=(12, 8))

    for drafter, values in accept_lengths.items():
        linestyle = '-' if 'mamba' in drafter.lower() else '--'  # Solid line for mamba, dashed line for pythia
        plt.plot(tasks, values, marker='o', label=drafter, linestyle=linestyle)

    # Customize the plot
    plt.xlabel('Tasks', fontsize=18)
    plt.ylabel('Average Accepted Tokens / Step', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Drafters', fontsize=14, title_fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot to the 'figures' folder in both SVG and PDF formats
    output_path_svg = os.path.join('figures', 'drafter_avg_accept_tokens.svg')
    output_path_pdf = os.path.join('figures', 'drafter_avg_accept_tokens.pdf')
    plt.savefig(output_path_svg)
    plt.savefig(output_path_pdf)
    print(f"Plot saved to {output_path_svg} and {output_path_pdf}")

    # Show the plot
    plt.show()


tasks = ['coding', 'math', 'rag', 'summarization', 'translation']
task_map = {
    'coding': 'coding',
    'math': 'math_reasoning',
    'rag': 'rag',
    'summarization': 'summarization',
    'translation': 'translation',
}
drafter_names = ['mamba-130m', 'mamba-370m', 'mamba-1.4b', 'pythia-160m', 'pythia-410m', 'pythia-1.4b']

# Handle __file__ safely in case you run this in a Jupyter Notebook
try:
    repo_root = Path(__file__).resolve().parents[1]
except NameError:
    repo_root = Path('.').resolve()

results_root = repo_root / 'results'
target_model = 'oasst-sft-4-pythia-12b-epoch-3.5'

accept_lengths = build_accept_lengths(results_root, target_model, task_map, drafter_names)
plot_avg_accept_tokens(tasks, drafter_names, accept_lengths)