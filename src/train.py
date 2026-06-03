import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Trains multiple models and returns a dictionary of trained models."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluates all trained models and returns a comparison DataFrame."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1
        })
        
    return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

def select_best_model(models: Dict[str, Any], evaluation_df: pd.DataFrame) -> Tuple[str, Any]:
    """Selects the best model based on accuracy from the evaluation DataFrame."""
    best_model_name = evaluation_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    return best_model_name, best_model
