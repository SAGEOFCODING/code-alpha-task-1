import streamlit as st
import pandas as pd
import os
import joblib

from src.predict import predict_species

# Must be the first streamlit command
st.set_page_config(page_title="Iris Classifier", page_icon="🌼", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Animated Bauhaus Style ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

    /* Main Container & Typography */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
        background-color: #f5f0e8;
        color: #1a1a1a;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Space Grotesk', sans-serif;
        color: #1a1a1a;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: -0.02em;
    }
    
    /* Global Streamlit elements */
    .stApp {
        background-color: #f5f0e8;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #ffcc00;
        color: #1a1a1a;
        border: 4px solid #1a1a1a !important;
        box-shadow: 6px 6px 0px rgba(26,26,26,1) !important;
        border-radius: 0px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.2s ease-out;
    }
    div.stButton > button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0px rgba(26,26,26,1) !important;
        color: #1a1a1a;
    }
    div.stButton > button:active {
        transform: translate(4px, 4px);
        box-shadow: 2px 2px 0px rgba(26,26,26,1) !important;
    }
    
    /* Sliders */
    div[data-baseweb="slider"] {
        padding-top: 10px;
    }
    div[data-testid="stThumbValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: bold;
        color: #1a1a1a;
        background: #ffcc00;
        border: 2px solid #1a1a1a;
        padding: 2px 6px;
    }

    /* Input Sections (Neo-brutalist panels) */
    .bauhaus-panel {
        background-color: #ffffff;
        border: 4px solid #1a1a1a;
        box-shadow: 4px 4px 0px rgba(26,26,26,1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s;
    }
    .bauhaus-panel:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0px rgba(26,26,26,1);
    }
    
    /* Native Streamlit Overrides for Bauhaus */
    div[data-testid="stDataFrame"] > div {
        border: 4px solid #1a1a1a !important;
        box-shadow: 4px 4px 0px rgba(26,26,26,1) !important;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stImage"] > img {
        border: 4px solid #1a1a1a !important;
        box-shadow: 4px 4px 0px rgba(26,26,26,1) !important;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stText"] {
        background-color: #ffffff;
        border: 4px solid #1a1a1a !important;
        box-shadow: 4px 4px 0px rgba(26,26,26,1) !important;
        padding: 1.5rem;
    }
    
    /* Result Card */
    .result-card {
        background-color: #ffffff;
        border: 4px solid #1a1a1a;
        box-shadow: 8px 8px 0px rgba(26,26,26,1);
        padding: 0;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    .result-header {
        background-color: #ffcc00;
        border-bottom: 4px solid #1a1a1a;
        padding: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 1.2rem;
    }
    .result-body {
        padding: 2rem;
        background-color: #faf7f2;
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    
    /* Geometric Shapes for Result Card */
    .result-shape-1 {
        position: absolute;
        top: 0;
        right: 0;
        width: 80px;
        height: 80px;
        background-color: #e63b2e;
        border-left: 4px solid #1a1a1a;
        border-bottom: 4px solid #1a1a1a;
        transform: translate(20px, -20px) rotate(45deg);
    }
    .result-shape-2 {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 60px;
        background-color: #0055ff;
        border-right: 4px solid #1a1a1a;
        border-top: 4px solid #1a1a1a;
        border-radius: 50%;
        transform: translate(-10px, 10px);
    }

    .result-species {
        background-color: #ffffff;
        border: 4px solid #1a1a1a;
        box-shadow: 4px 4px 0px rgba(26,26,26,1);
        padding: 1.5rem;
        margin-top: 2rem;
        position: relative;
        z-index: 10;
        transition: transform 0.3s;
    }
    .result-species:hover {
        transform: translateY(-4px);
        box-shadow: 8px 8px 0px rgba(26,26,26,1);
    }
    .result-badge {
        position: absolute;
        top: -15px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #ffcc00;
        border: 2px solid #1a1a1a;
        padding: 2px 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .result-value {
        color: #1a1a1a;
        font-size: 2.2rem;
        font-weight: bold;
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        margin: 0;
    }
    .result-confidence {
        display: inline-block;
        background-color: #e63b2e;
        color: #ffffff;
        border: 2px solid #1a1a1a;
        box-shadow: 2px 2px 0px rgba(26,26,26,1);
        padding: 4px 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: bold;
        margin-top: 1rem;
        text-transform: uppercase;
    }
    
    /* Override native metric containers */
    div[data-testid="metric-container"] {
        display: none; /* We will use custom HTML instead for KPIs */
    }
    
    /* Custom KPI Cards */
    .kpi-card {
        background-color: #ffffff;
        border: 4px solid #1a1a1a;
        box-shadow: 6px 6px 0px rgba(26,26,26,1);
        padding: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.3s;
    }
    .kpi-card:hover {
        transform: translate(-4px, -4px);
        box-shadow: 10px 10px 0px rgba(26,26,26,1);
    }
    .kpi-icon {
        width: 50px;
        height: 50px;
        border: 2px solid #1a1a1a;
        box-shadow: 2px 2px 0px rgba(26,26,26,1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .kpi-icon.yellow { background-color: #ffcc00; }
    .kpi-icon.red { background-color: #e63b2e; color: white; }
    .kpi-icon.blue { background-color: #0055ff; color: white; }
    
    .kpi-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: bold;
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #1a1a1a;
        margin: 0;
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: bold;
        font-size: 1.8rem;
        color: #1a1a1a;
        margin: 0;
        line-height: 1;
    }
    
    /* DataFrame Styling */
    .dataframe {
        border: 4px solid #1a1a1a !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 4px solid #1a1a1a;
    }
    
    /* Home Page Enhancements */
    .bauhaus-list-item {
        background: #f5f0e8; 
        border: 4px solid #1a1a1a; 
        padding: 1rem; 
        margin-bottom: 1rem; 
        box-shadow: 4px 4px 0px rgba(26,26,26,1); 
        transition: all 0.2s ease-out;
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }
    .bauhaus-list-item:hover {
        transform: translate(-4px, -4px);
        box-shadow: 8px 8px 0px rgba(26,26,26,1);
    }
    .bauhaus-badge {
        font-family: 'Space Grotesk', sans-serif; 
        border: 2px solid #1a1a1a; 
        padding: 4px 12px; 
        font-weight: bold; 
        margin-right: 10px;
        box-shadow: 2px 2px 0px rgba(26,26,26,1);
        text-transform: uppercase;
        display: inline-block;
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
    st.sidebar.title("🌼 Iris Classifier")
    st.sidebar.markdown("**Digital Herbarium v1.0**")
    st.sidebar.markdown("---")
    
    # Navigation
    menu = ["Home", "Data Exploration", "Model Performance", "Prediction"]
    choice = st.sidebar.radio("Go to", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.info("High-performance machine learning model for Iris flower classification using Neo-Brutalism aesthetics.")
    
    if choice == "Home":
        st.title("Iris Classification Engine")
        st.markdown("""
<div class="bauhaus-panel" style="position:relative; overflow:hidden; background-image: radial-gradient(#1a1a1a 1.5px, transparent 1.5px); background-size: 20px 20px; background-color: #ffffff;">
<div style="position:absolute; top:-20px; right:-20px; width:120px; height:120px; background:#ffcc00; border-radius:50%; border:4px solid #1a1a1a;"></div>
<div style="position:absolute; bottom:30px; right:60px; width:50px; height:50px; background:#e63b2e; transform:rotate(45deg); border:4px solid #1a1a1a;"></div>
<div style="position:absolute; top:40%; right:10px; width:30px; height:30px; background:#0055ff; border:4px solid #1a1a1a;"></div>
<div style="background:#ffffff; border:4px solid #1a1a1a; padding:1.5rem; display:inline-block; margin-bottom:1.5rem; box-shadow:6px 6px 0px #1a1a1a; position:relative; z-index:1;">
<h2 style="margin:0; font-size:2.2rem;">WELCOME TO THE <span style="color:#e63b2e;">DIGITAL HERBARIUM</span></h2>
</div>
<div style="background:#ffffff; border:4px solid #1a1a1a; padding:1.5rem; max-width:85%; position:relative; z-index:1; box-shadow:4px 4px 0px #1a1a1a; margin-bottom: 2rem;">
<p style="font-size:1.1rem; margin:0; font-weight:500;">This dashboard interfaces with a locally trained machine learning model to classify Iris flowers based on precise botanical measurements.</p>
</div>
<div style="position:relative; z-index:1; max-width: 90%;">
<div class="bauhaus-list-item">
<div style="margin-bottom: 0.5rem;">
<span class="bauhaus-badge" style="background:#0055ff; color:white;">1. Data Exploration</span>
</div>
Review raw dataset and statistical distributions. All visualizations generated during EDA are available here.
</div>
<div class="bauhaus-list-item">
<div style="margin-bottom: 0.5rem;">
<span class="bauhaus-badge" style="background:#e63b2e; color:white;">2. Model Performance</span>
</div>
Analyze the accuracy and confusion matrix of the trained model.
</div>
<div class="bauhaus-list-item">
<div style="margin-bottom: 0.5rem;">
<span class="bauhaus-badge" style="background:#ffcc00; color:#1a1a1a;">3. Prediction</span>
</div>
Input new flower measurements for real-time inference.
</div>
</div>
</div>
""", unsafe_allow_html=True)
        
    elif choice == "Data Exploration":
        st.title("Data Exploration & Visualizations")
        df = load_dataset()
        
        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Dataset Visualizations")
            
            # Row 1: Pairplot and Heatmap
            col1, col2 = st.columns(2)
            try:
                with col1:
                    st.image(os.path.join("reports", "assets", "pairplot.png"), use_container_width=True)
                with col2:
                    st.image(os.path.join("reports", "assets", "correlation_heatmap.png"), use_container_width=True)
            except Exception:
                st.warning("Could not load Pairplot or Heatmap. Please ensure training pipeline was run.")
                
            # Row 2: Histograms and Violin Plots
            col3, col4 = st.columns(2)
            try:
                with col3:
                    st.image(os.path.join("reports", "assets", "histograms.png"), use_container_width=True)
                with col4:
                    st.image(os.path.join("reports", "assets", "violin_plots.png"), use_container_width=True)
            except Exception:
                st.warning("Could not load Histograms or Violin Plots. Please ensure training pipeline was run.")
                
        else:
            st.error("Dataset not found. Please ensure `data/iris.csv` exists.")
            
    elif choice == "Model Performance":
        st.title("Model Performance")
        
        try:
            eval_df = pd.read_csv(os.path.join("reports", "evaluation_summary.csv"))
            
            st.subheader("Model Comparison")
            st.dataframe(eval_df.style.highlight_max(subset=['Accuracy', 'F1 Score'], color='#ffcc00'), use_container_width=True)
            
            best_model_name = eval_df.iloc[0]['Model']
            st.markdown(f"<div class='bauhaus-panel' style='padding:1rem; margin-top:1rem;'><strong>Active Model:</strong> {best_model_name} (Accuracy: {eval_df.iloc[0]['Accuracy']:.4f})</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<div style='padding:1rem; border:4px solid #1a1a1a; border-bottom:0; background:#ffcc00; font-family:Space Grotesk; font-weight:bold;'>Confusion Matrix ({best_model_name})</div>", unsafe_allow_html=True)
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
        st.title("Live Prediction Inference")
        
        scaler, label_encoder, feature_names, best_model = load_models_and_objects()
        
        if scaler is None:
            st.error("Models not found. Please run the training pipeline first (`python main.py`).")
            return
            
        # Layout: 2 Columns for Input and Output
        main_col1, main_col2 = st.columns([1, 1], gap="large")
        
        with main_col1:
            st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 4px solid #1a1a1a; padding-bottom:1rem; margin-bottom:1rem;">
                <h2 style="margin:0;">Input Parameters</h2>
                <span style="background:#ffcc00; border:2px solid #1a1a1a; font-family:'Space Grotesk'; font-weight:bold; font-size:0.8rem; padding:4px 8px; box-shadow:2px 2px 0px #1a1a1a; text-transform:uppercase;">Live Inference</span>
            </div>
            """, unsafe_allow_html=True)
            
            sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            sepal_width = st.slider("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("Run Classification", type="primary", use_container_width=True)
            
        with main_col2:
            if predict_btn:
                features = [sepal_length, sepal_width, petal_length, petal_width]
                predicted_species, prob_dict = predict_species(best_model, scaler, label_encoder, features, feature_names)
                
                max_prob = max(prob_dict.values()) * 100
                
                st.markdown(f"""
<div class="result-card">
<div class="result-header">Classification Result</div>
<div class="result-body">
<div class="result-shape-1"></div>
<div class="result-shape-2"></div>
<div class="result-species">
<div class="result-badge">Predicted Species</div>
<h3 class="result-value">{predicted_species}</h3>
<div class="result-confidence">Confidence: {max_prob:.2f}%</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
                
            else:
                st.markdown("""
<div class="result-card">
<div class="result-header">Classification Result</div>
<div class="result-body">
<div class="result-shape-1" style="background-color:#d0cbc3;"></div>
<div class="result-species" style="box-shadow:none;">
<div class="result-badge" style="background:#e8e3da;">Awaiting Input</div>
<h3 class="result-value" style="color:#a0a0a0;">---</h3>
<div style="font-family:'Space Grotesk'; font-weight:bold; margin-top:1rem; text-transform:uppercase; color:#a0a0a0;">Adjust parameters and click 'Run'</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
                
        # Bottom Row: KPIs
        st.markdown("<br><hr style='border-top: 4px solid #1a1a1a;'><br>", unsafe_allow_html=True)
        
        # Load accuracy for KPI if available
        acc = "N/A"
        f1 = "N/A"
        try:
            eval_df = pd.read_csv(os.path.join("reports", "evaluation_summary.csv"))
            acc = f"{eval_df.iloc[0]['Accuracy']*100:.2f}%"
            f1 = f"{eval_df.iloc[0]['F1 Score']:.3f}"
        except Exception:
            pass
            
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon yellow">🎯</div>
                <div>
                    <p class="kpi-label">Model Accuracy</p>
                    <p class="kpi-value">{acc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon red">📈</div>
                <div>
                    <p class="kpi-label">F1 Score</p>
                    <p class="kpi-value">{f1}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon blue">📊</div>
                <div>
                    <p class="kpi-label">Total Samples</p>
                    <p class="kpi-value">150</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
