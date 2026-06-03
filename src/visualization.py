import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set modern plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.autolayout': True})

def save_plot(fig, filename: str, output_dir: str = "reports/assets"):
    """Saves a matplotlib figure to the given directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "reports/assets"):
    """Generates and saves a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16)
    save_plot(fig, "correlation_heatmap.png", output_dir)

def plot_pairplot(df: pd.DataFrame, target_col: str = 'Species', output_dir: str = "reports/assets"):
    """Generates and saves a pairplot."""
    pair_plot = sns.pairplot(df, hue=target_col, palette="husl", markers=["o", "s", "D"])
    pair_plot.fig.suptitle("Pairplot of Features", y=1.02, fontsize=16)
    save_plot(pair_plot.fig, "pairplot.png", output_dir)

def plot_distributions(df: pd.DataFrame, target_col: str = 'Species', output_dir: str = "reports/assets"):
    """Generates and saves histograms and violin plots for features."""
    features = df.drop(target_col, axis=1).columns
    
    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue=target_col, kde=True, ax=axes[i//2, i%2], palette="husl")
        axes[i//2, i%2].set_title(f"Distribution of {feature}")
    fig.suptitle("Feature Distributions", fontsize=16)
    save_plot(fig, "histograms.png", output_dir)
    
    # Violin plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, feature in enumerate(features):
        sns.violinplot(data=df, x=target_col, y=feature, ax=axes[i//2, i%2], palette="husl")
        axes[i//2, i%2].set_title(f"Violin Plot of {feature}")
    fig.suptitle("Feature Violin Plots by Species", fontsize=16)
    save_plot(fig, "violin_plots.png", output_dir)

def generate_all_visualizations(df: pd.DataFrame, target_col: str = 'Species', output_dir: str = "reports/assets"):
    """Runs all visualization functions."""
    plot_correlation_heatmap(df, output_dir)
    plot_pairplot(df, target_col, output_dir)
    plot_distributions(df, target_col, output_dir)
