import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_results():
    # Load all evaluation CSVs
    files = glob.glob('results/*_eval.csv')
    if not files:
        print("No evaluation CSVs found in results/ directory.")
        return

    summary_data = []

    # Process each file to create the summary table
    for f in files:
        model_name = os.path.basename(f).replace('_eval.csv', '')
        df = pd.read_csv(f)
        
        summary_data.append({
            'Model': model_name,
            'Completion Rate': df['flag'].mean(),
            'Avg Reward': df['reward'].mean(),
            'Avg Max X': df['max_x'].mean(),
            'Avg Steps': df['steps'].mean(),
            'Best Max X': df['max_x'].max()
        })

    summary_df = pd.DataFrame(summary_data)
    
    # Save the results table
    summary_df.to_csv('results/summary_table.csv', index=False)
    print("Saved summary table to results/summary_table.csv")

    # Determine colors for PPO vs DQN
    colors = ['orange' if 'ppo' in m.lower() else 'blue' for m in summary_df['Model']]

    # Plot 1: Average Reward Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Model'], summary_df['Avg Reward'], color=colors)
    plt.title('Average Reward: PPO vs DQN')
    plt.ylabel('Reward')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/reward_comparison.png')
    
    # Plot 2: Max X-Position Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Model'], summary_df['Avg Max X'], color=colors)
    plt.title('Average Max X-Position: PPO vs DQN')
    plt.ylabel('Distance (Max X)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/max_x_comparison.png')
    print("Saved comparison plots to results/")

if __name__ == "__main__":
    analyze_results()