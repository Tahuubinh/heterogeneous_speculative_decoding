import matplotlib.pyplot as plt
import os

def plot_drafter_throughput(tasks, drafter_names, throughputs):
    """
    Create a line plot for drafter throughputs compared to the target model and save it to the 'figures' folder.

    Parameters:
        tasks (list of str): List of task names.
        drafter_names (list of str): List of drafter names.
        throughputs (dict): Dictionary where keys are drafter names and values are lists of throughput ratios for each task.
    """
    # Ensure the 'figures' folder exists
    os.makedirs('figures', exist_ok=True)

    # Create the line plot
    plt.figure(figsize=(12, 8))

    for drafter, ratios in throughputs.items():
        linestyle = '-' if 'mamba' in drafter.lower() else '--'  # Solid line for mamba, dashed line for pythia
        plt.plot(tasks, ratios, marker='o', label=drafter, linestyle=linestyle)

    # Customize the plot
    plt.xlabel('Tasks', fontsize=18)
    plt.ylabel('Throughput Ratio (Speculative Decoding / Online Target)', fontsize=18)
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

# Example usage (you can modify these lists and dictionary as needed)
tasks = ['coding', 'math', 'rag', 'summarization', 'translation']
drafter_names = ['mamba-130m', 'mamba-370m', 'mamba-1.4b', 'pythia-160m', 'pythia-410m', 'pythia-1.4b']
throughputs = {
    'mamba-130m': [1.2, 1.1, 1.3, 1.4, 1.6],
    'mamba-370m': [1.1, 1.0, 1.2, 1.3, 1.1],
    'mamba-1.4b': [1.3, 1.2, 1.4, 1.5, 1.3],
    'pythia-160m': [1.0, 0.9, 1.1, 1.2, 1.0],
    'pythia-410m': [1.5, 1.3, 1.5, 1.1, 1.5],
    'pythia-1.4b': [1.7, 1.5, 1.6, 1.8, 1.2]
}

plot_drafter_throughput(tasks, drafter_names, throughputs)