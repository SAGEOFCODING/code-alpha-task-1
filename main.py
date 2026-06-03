import os
import sys
import logging
from src.data_loader import load_data, clean_data
from src.preprocessing import preprocess_data
from src.visualization import generate_all_visualizations
from src.train import train_models, evaluate_models, select_best_model
from src.evaluate import generate_classification_report, plot_confusion_matrix
from src.utils import save_object

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Iris Flower Classification Pipeline")
    
    # 1. Load Data
    data_path = os.path.join("data", "iris.csv")
    logging.info(f"Loading dataset from {data_path}")
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(1)
        
    logging.info("Cleaning data...")
    df = clean_data(df)
    
    # 2. EDA & Visualizations
    logging.info("Generating visualizations...")
    generate_all_visualizations(df, target_col='Species')
    
    # 3. Preprocessing
    logging.info("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(df, target_col='Species')
    
    # Save preprocessing objects
    logging.info("Saving preprocessing objects...")
    save_object(scaler, "scaler.joblib")
    save_object(label_encoder, "label_encoder.joblib")
    
    # Save feature names for later use
    feature_names = df.drop('Species', axis=1).columns.tolist()
    save_object(feature_names, "feature_names.joblib")
    
    # 4. Training
    logging.info("Training models...")
    models = train_models(X_train, y_train)
    
    # 5. Evaluation
    logging.info("Evaluating models...")
    evaluation_df = evaluate_models(models, X_test, y_test)
    logging.info(f"Model Evaluation Results:\n{evaluation_df}")
    
    # Save evaluation summary
    evaluation_df.to_csv(os.path.join("reports", "evaluation_summary.csv"), index=False)
    
    # 6. Select Best Model
    best_model_name, best_model = select_best_model(models, evaluation_df)
    logging.info(f"Best model selected: {best_model_name}")
    
    # Save Best Model
    save_object(best_model, "best_model.joblib")
    
    # Generate Confusion Matrix and Classification Report for Best Model
    logging.info("Generating reports for the best model...")
    plot_confusion_matrix(best_model, X_test, y_test, target_names=label_encoder.classes_, model_name=best_model_name)
    report = generate_classification_report(best_model, X_test, y_test, target_names=label_encoder.classes_)
    
    with open(os.path.join("reports", "classification_report.txt"), "w") as f:
        f.write(report)
        
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
