import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def generate_classification_report(model, X_test: pd.DataFrame, y_test: pd.Series, target_names: list) -> str:
    """Generates a text classification report."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)

def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, target_names: list, model_name: str, output_dir: str = "reports/assets"):
    """Plots and saves the confusion matrix for a given model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath
