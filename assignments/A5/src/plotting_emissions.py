import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def read_csv_files():
    """Read CSV files from the 'in' folder."""
    a1_df = pd.read_csv(os.path.join("in", "A1_emissions.csv"))
    a2_logreg_df = pd.read_csv(os.path.join("in", "A2_logreg_emissions.csv"))
    a2_mlp_df = pd.read_csv(os.path.join("in", "A2_mlp_emissions.csv"))
    a3_df = pd.read_csv(os.path.join("in", "A3_emissions.csv"))
    a4_df = pd.read_csv(os.path.join("in", "A4_emissions.csv"))
    return a1_df, a2_logreg_df, a2_mlp_df, a3_df, a4_df

def calculate_total_emissions(a1_df, a2_logreg_df, a2_mlp_df, a3_df, a4_df):
    """Calculate total emissions for each assignment."""
    total_emissions = {
        "Assignment 1": a1_df["emissions"].sum(),
        "Assignment 2 - logreg": a2_logreg_df["emissions"].sum(),
        "Assignment 2 - mlp": a2_mlp_df["emissions"].sum(),
        "Assignment 3": a3_df["emissions"].sum(),
        "Assignment 4": a4_df["emissions"].sum()
    }
    return total_emissions

def plot_total_emissions(assignments, emissions):
    """Plot total emissions for each assignment."""
    assignment_colors = {
        "Assignment 1": 'skyblue',
        "Assignment 2 - logreg": 'red',
        "Assignment 2 - mlp": 'green',
        "Assignment 3": 'royalblue',
        "Assignment 4": 'gold'
    }
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(assignments, emissions)
    for i, bar in enumerate(bars):
        assignment = assignments[i]
        bar.set_color(assignment_colors.get(assignment, 'skyblue'))
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%.2f%%' % (emissions[i] / sum(emissions) * 100),
                 ha='center', va='bottom', fontsize=10, rotation=0)
    plt.title('Total Emissions for Each Assignment', fontsize=14, fontweight='bold')
    plt.xlabel('Assignment')
    plt.ylabel('Total Emissions (CO₂eq), logscaled')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("out", "total_emissions.png"))

def group_and_plot_task_emissions(a2_logreg_df, a2_mlp_df, a3_df, a4_df):
    """Group tasks by name and sum their emissions for each assignment."""
    assignments = {
        "Assignment 2 - logreg": a2_logreg_df,
        "Assignment 2 - mlp": a2_mlp_df,
        "Assignment 3": a3_df,
        "Assignment 4": a4_df
    }
    
    task_emissions = {}
    for assignment_name, df in assignments.items():
        task_emissions[assignment_name] = df.groupby("task_name")["emissions"].sum()
    
    plt.figure(figsize=(12, 8))
    
    for assignment_name, task_emission in task_emissions.items():
        color = 'r' if 'logreg' in assignment_name else 'g' if 'mlp' in assignment_name else 'b' if '3' in assignment_name else 'y'
        for i, (task, emission) in enumerate(zip(task_emission.index, task_emission.values)):
            plt.stem([task], [emission], markerfmt='o', linefmt=f'{color}-', basefmt=' ', label=assignment_name if i == 0 else None)
            plt.text(task, emission, '%.2f%%' % (emission / sum(task_emission.values) * 100), ha='right' if color != 'r' else 'left', va='bottom', fontsize=10, color=color)
    
    plt.title('Emissions for Each Task in Each Assignment', fontsize=14, fontweight='bold')
    plt.xlabel('Task')
    plt.ylabel('Total Emissions (CO₂eq), logscale')
    plt.xticks(rotation=90)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("out", "task_emissions.png"))

def main():
    """Main function."""
    if not os.path.exists("out"):
        os.makedirs("out")
    
    a1_df, a2_logreg_df, a2_mlp_df, a3_df, a4_df = read_csv_files()
    total_emissions = calculate_total_emissions(a1_df, a2_logreg_df, a2_mlp_df, a3_df, a4_df)
    assignments = list(total_emissions.keys())
    emissions = list(total_emissions.values())
    plot_total_emissions(assignments, emissions)
    group_and_plot_task_emissions(a2_logreg_df, a2_mlp_df, a3_df, a4_df)

if __name__ == "__main__":
    main()
