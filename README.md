# Iris Flower Classification Project 🌿

A complete, professional-grade Machine Learning project for Iris Flower classification using Python, scikit-learn, and Streamlit.
## Overview

This project demonstrates an end-to-end Machine Learning pipeline that classifies the classic Iris dataset. The system features the **Botanical Intelligence Dark System**—a custom-designed, high-fidelity UI aesthetic implemented natively in Streamlit to provide a sleek, modern, and immersive user experience.

### Key Features
- **End-to-End Pipeline**: Includes automated data loading, cleaning, and exploratory data analysis (EDA).
- **Model Comparison**: Trains and evaluates Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machine, automatically persisting the best performer.
- **Botanical Intelligence UI**: A beautifully crafted Dark Mode dashboard with deep navy surfaces and electric violet accents.
- **Interactive Inference**: Real-time predictions via interactive sliders on the Streamlit dashboard, complete with dynamic confidence scores and model diagnostic KPIs.
- **Comprehensive Visualizations**: Automatically generated feature distributions (Violin Plots, Histograms) and correlation matrices natively embedded within the UI.

## Project Structure

```text
iris-classification/
├── data/
│   └── iris.csv              # The Iris dataset
├── models/                   # Serialized ML models and preprocessors (.joblib)
├── reports/
│   ├── assets/               # Generated EDA visualizations and UI assets
│   ├── evaluation_summary.csv
│   └── classification_report.txt
├── src/                      # Modular source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── .streamlit/               # UI Configuration overrides
│   └── config.toml
├── app.py                    # The Streamlit dashboard
├── main.py                   # Main orchestrator to train the ML pipeline
└── requirements.txt          # Python dependencies
```

## Setup Instructions

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models:**
   Run the pipeline to train models, generate visualizations, and save the best model to the `models/` directory.
   ```bash
   python main.py
   ```

4. **Launch the Dashboard:**
   Fire up the Botanical Intelligence UI.
   ```bash
   streamlit run app.py
   ```