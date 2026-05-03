import matplotlib.pyplot as plt
import os

def plot_model_throughput(model_names, throughputs):
    """
    Create a bar plot for model throughputs and save it to the 'figures' folder.

    Parameters:
        model_names (list of str): List of model names.
        throughputs (list of float): List of throughput values corresponding to the models.
    """
    # Ensure the 'figures' folder exists
    os.makedirs('figures', exist_ok=True)

    # Define colors for specific models
    colors = ['#1a80bb' if 'mamba' in name.lower() else '#a00000' for name in model_names]

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_names, throughputs, color=colors, edgecolor='black')

    # Add throughput values on top of each bar
    for bar, throughput in zip(bars, throughputs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of the bar)
            bar.get_height() + 5,  # Y-coordinate (slightly above the bar)
            f'{throughput:.2f}',  # Text to display (formatted to 2 decimal places)
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=18  # Larger font size
        )

    # Adjust y-axis limit to provide more space above the tallest bar
    plt.ylim(0, max(throughputs) + 50)

    # Customize the plot
    plt.xlabel('Models', fontsize=18)  # Update x-axis label to 'Models'
    plt.ylabel('Throughput (tokens/s)', fontsize=18)  # Update y-axis label to 'Throughput (tokens/s)'
    plt.xticks(rotation=45, ha='right', fontsize=18)  # Rotate x-axis labels and keep font size
    plt.yticks(fontsize=18)  # Keep font size for y-axis labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
    plt.tight_layout()

    # Save the plot to the 'figures' folder in both PNG, SVG, and PDF formats
    # output_path_png = os.path.join('figures', 'model_throughput.png')
    output_path_svg = os.path.join('figures', 'model_throughput.svg')
    output_path_pdf = os.path.join('figures', 'model_throughput.pdf')
    # plt.savefig(output_path_png)
    plt.savefig(output_path_svg)
    plt.savefig(output_path_pdf)
    # print(f"Plot saved to {output_path_png}, {output_path_svg}, and {output_path_pdf}")

    # Show the plot
    plt.show()

# Example usage (you can modify these lists as needed)
model_names = ['mamba-130m', 'mamba-370m', 'mamba-1.4b', 'pythia-160m', 'pythia-410m', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b']
throughputs = [775.36, 325.96, 109.81, 92.79, 48.40, 46.38, 34.66, 19.31]

plot_model_throughput(model_names, throughputs)