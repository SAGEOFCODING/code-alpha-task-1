import streamlit as st
import pandas as pd
import os
import joblib

from src.predict import predict_species

# Must be the first streamlit command
st.set_page_config(page_title="Iris Classifier", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Botanical Intelligence Dark Mode ---
st.markdown("""
<style>
    /* Main Container & Typography */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Manrope', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Epilogue', sans-serif;
        color: #f8fafc;
    }
    
    /* Cards and KPI Containers */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 5% 5% 5% 10%;
        border-radius: 16px;
        color: #dae2fd;
        overflow-wrap: break-word;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
    }
    
    /* Custom Result Card */
    .result-card {
        background-color: #1e293b;
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .result-title {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .result-value {
        color: #a855f7;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        font-family: 'Epilogue', sans-serif;
    }
    .result-confidence {
        color: #10b981;
        font-size: 1.1rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* DataFrame Styling */
    .dataframe {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_objects():
    try:
        scaler = joblib.load(os.path.join("models", "scaler.joblib"))
        label_encoder = joblib.load(os.path.join("models", "label_encoder.joblib"))
        feature_names = joblib.load(os.path.join("models", "feature_names.joblib"))
        best_model = joblib.load(os.path.join("models", "best_model.joblib"))
        return scaler, label_encoder, feature_names, best_model
    except FileNotFoundError:
        return None, None, None, None

def load_dataset():
    try:
        df = pd.read_csv(os.path.join("data", "iris.csv"))
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        return df
    except FileNotFoundError:
        return None

def main():
    st.sidebar.title("🌿 Botanical Intelligence")
    st.sidebar.markdown("Navigate through the application sections.")
    
    # Navigation
    menu = ["Home", "Data Exploration", "Model Performance", "Prediction"]
    choice = st.sidebar.radio("Go to", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.info("High-performance machine learning model for Iris flower classification.")
    
    if choice == "Home":
        st.title("Iris Classification Engine")
        st.markdown("""
        Welcome to the **Botanical Intelligence System**.
        
        This dashboard interfaces with a locally trained machine learning model to classify Iris flowers based on precise botanical measurements.
        
        1. **Data Exploration**: Review raw dataset and statistical distributions. All visualizations generated during EDA are available here.
        2. **Model Performance**: Analyze the accuracy and confusion matrix of the trained model.
        3. **Prediction**: Input new flower measurements for real-time inference.
        """)
        
    elif choice == "Data Exploration":
        st.title("Data Exploration & Visualizations")
        df = load_dataset()
        
        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Dataset Visualizations")
            st.markdown("Here are the comprehensive visual distributions of the Iris dataset.")
            
            # Row 1: Pairplot and Heatmap
            col1, col2 = st.columns(2)
            try:
                with col1:
                    st.image(os.path.join("reports", "assets", "pairplot.png"), caption="Pairplot: Feature Relationships", use_container_width=True)
                with col2:
                    st.image(os.path.join("reports", "assets", "correlation_heatmap.png"), caption="Correlation Heatmap", use_container_width=True)
            except Exception:
                st.warning("Could not load Pairplot or Heatmap. Please ensure training pipeline was run.")
                
            # Row 2: Histograms and Violin Plots
            col3, col4 = st.columns(2)
            try:
                with col3:
                    st.image(os.path.join("reports", "assets", "histograms.png"), caption="Histograms: Feature Distributions", use_container_width=True)
                with col4:
                    st.image(os.path.join("reports", "assets", "violin_plots.png"), caption="Violin Plots: Data Density", use_container_width=True)
            except Exception:
                st.warning("Could not load Histograms or Violin Plots. Please ensure training pipeline was run.")
                
        else:
            st.error("Dataset not found. Please ensure `data/iris.csv` exists.")
            
    elif choice == "Model Performance":
        st.title("Model Performance")
        
        try:
            eval_df = pd.read_csv(os.path.join("reports", "evaluation_summary.csv"))
            st.subheader("Model Comparison")
            
            # Use styling for better presentation
            st.dataframe(eval_df.style.highlight_max(subset=['Accuracy', 'F1 Score'], color='rgba(168, 85, 247, 0.3)'), use_container_width=True)
            
            best_model_name = eval_df.iloc[0]['Model']
            st.success(f"**Active Model:** {best_model_name} (Accuracy: {eval_df.iloc[0]['Accuracy']:.4f})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Confusion Matrix ({best_model_name})")
                cm_path = os.path.join("reports", "assets", f"confusion_matrix_{best_model_name.replace(' ', '_').lower()}.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, use_container_width=True)
                else:
                    import glob
                    cm_files = glob.glob(os.path.join("reports", "assets", "confusion_matrix_*.png"))
                    if cm_files:
                        st.image(cm_files[0], use_container_width=True)
            
            with col2:
                st.subheader("Classification Report")
                try:
                    with open(os.path.join("reports", "classification_report.txt"), "r") as f:
                        report = f.read()
                    st.text(report)
                except FileNotFoundError:
                    st.warning("Classification report not found.")
                
        except FileNotFoundError:
            st.error("Evaluation results not found.")
            
    elif choice == "Prediction":
        st.title("Live Prediction Interface")
        
        scaler, label_encoder, feature_names, best_model = load_models_and_objects()
        
        if scaler is None:
            st.error("Models not found. Please run the training pipeline first (`python main.py`).")
            return
            
        # Layout: 2 Columns for Input and Output
        main_col1, main_col2 = st.columns([1, 1], gap="large")
        
        with main_col1:
            st.subheader("Input Parameters")
            st.markdown("Adjust the sliders to input botanical measurements.")
            
            sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            sepal_width = st.slider("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            
            predict_btn = st.button("Run Inference", type="primary", use_container_width=True)
            
        with main_col2:
            st.subheader("Classification Result")
            
            if predict_btn:
                features = [sepal_length, sepal_width, petal_length, petal_width]
                predicted_species, prob_dict = predict_species(best_model, scaler, label_encoder, features, feature_names)
                
                max_prob = max(prob_dict.values()) * 100
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Predicted Species</div>
                    <div class="result-value">{predicted_species}</div>
                    <div class="result-confidence">Confidence Score: {max_prob:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="result-card" style="border-color: rgba(255,255,255,0.05);">
                    <div class="result-title">Awaiting Input</div>
                    <div style="color: #64748b; margin-top: 1rem;">Adjust parameters and click 'Run Inference'</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Bottom Row: KPIs
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
        st.subheader("Model Diagnostics")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        
        # Load accuracy for KPI if available
        acc = "N/A"
        f1 = "N/A"
        try:
            eval_df = pd.read_csv(os.path.join("reports", "evaluation_summary.csv"))
            acc = f"{eval_df.iloc[0]['Accuracy']*100:.2f}%"
            f1 = f"{eval_df.iloc[0]['F1 Score']:.3f}"
        except Exception:
            pass
            
        with kpi1:
            st.metric(label="Model Accuracy", value=acc)
        with kpi2:
            st.metric(label="F1 Score", value=f1)
        with kpi3:
            st.metric(label="Total Samples", value="150")

if __name__ == "__main__":
    main()
