import matplotlib.pyplot as plt
import os
import json
import math
from pathlib import Path

def load_throughput_tokens_per_sec(metrics_path):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    metrics = payload.get('metrics', {})
    value = metrics.get('throughput_tokens_per_sec')
    if value is None:
        raise KeyError(f"throughput_tokens_per_sec not found in {metrics_path}")
    return float(value)


def build_speedup(results_root, target_model, task_map, drafter_names):
    baseline_dir = Path(results_root) / target_model / 'baseline' / 'none'

    baseline_throughput = {}
    for task_label, task_dir in task_map.items():
        baseline_metrics = baseline_dir / task_dir / 'baseline' / 'metrics.json'
        baseline_throughput[task_label] = load_throughput_tokens_per_sec(baseline_metrics)

    speedup = {}
    for drafter in drafter_names:
        backend_dir = 'mamba-replay' if 'mamba' in drafter.lower() else 'transformer'
        drafter_dir = Path(results_root) / target_model / drafter / 'adaptive'

        ratios = []
        for task_label, task_dir in task_map.items():
            spec_metrics = drafter_dir / task_dir / backend_dir / 'metrics.json'
            if not spec_metrics.exists():
                print(f"Warning: missing file {spec_metrics}")
                ratios.append(math.nan)
                continue

            spec_tp = load_throughput_tokens_per_sec(spec_metrics)
            base_tp = baseline_throughput[task_label]
            ratios.append(spec_tp / base_tp)

        speedup[drafter] = ratios

    return speedup


def plot_drafter_throughput(tasks, drafter_names, speedup):
    """
    Create a line plot for drafter speedup compared to target-only baseline and save it to the 'figures' folder.

    Parameters:
        tasks (list of str): List of task names.
        drafter_names (list of str): List of drafter names.
        speedup (dict): Dictionary where keys are drafter names and values are lists of speedup ratios for each task.
    """
    # Ensure the 'figures' folder exists
    os.makedirs('figures', exist_ok=True)

    # Create the line plot
    plt.figure(figsize=(12, 8))

    for drafter, ratios in speedup.items():
        linestyle = '-' if 'mamba' in drafter.lower() else '--'  # Solid line for mamba, dashed line for pythia
        plt.plot(tasks, ratios, marker='o', label=drafter, linestyle=linestyle)

    # Customize the plot
    plt.xlabel('Tasks', fontsize=18)
    plt.ylabel('Speedup', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Drafters', fontsize=14, title_fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot to the 'figures' folder in both SVG and PDF formats
    output_path_svg = os.path.join('figures', 'drafter_throughput_comparison.svg')
    output_path_pdf = os.path.join('figures', 'drafter_throughput_comparison.pdf')
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

repo_root = Path(__file__).resolve().parents[1]
results_root = repo_root / 'results'
target_model = 'oasst-sft-4-pythia-12b-epoch-3.5'

speedup = build_speedup(results_root, target_model, task_map, drafter_names)
plot_drafter_throughput(tasks, drafter_names, speedup)