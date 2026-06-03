import pandas as pd
from typing import Tuple, List, Dict
import numpy as np

def predict_species(model, scaler, label_encoder, features: List[float], feature_names: List[str]) -> Tuple[str, Dict[str, float]]:
    """
    Predicts the Iris species given the features.
    Returns the predicted species name and a dictionary of class probabilities.
    """
    # Create DataFrame to maintain feature names
    input_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale features
    scaled_features = scaler.transform(input_df)
    
    # Convert back to DataFrame to prevent warning messages from sklearn
    scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
    
    # Predict
    prediction_encoded = model.predict(scaled_df)
    probabilities = model.predict_proba(scaled_df)[0]
    
    # Decode
    predicted_species = label_encoder.inverse_transform(prediction_encoded)[0]
    
    # Map probabilities to class names
    classes = label_encoder.classes_
    prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    
    return predicted_species, prob_dict
