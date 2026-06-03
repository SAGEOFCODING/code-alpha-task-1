import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Any

def preprocess_data(df: pd.DataFrame, target_col: str = 'Species') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, LabelEncoder]:
    """
    Separates features and targets, scales features, and encodes targets.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Convert target arrays to pandas Series for consistency
    y_train_series = pd.Series(y_train, name=target_col)
    y_test_series = pd.Series(y_test, name=target_col)
    
    return X_train_scaled, X_test_scaled, y_train_series, y_test_series, scaler, le
