import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Try importing ML libraries with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except ImportError:
    st.error("TensorFlow not installed. Please install: pip install tensorflow")
    TF_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.error("Transformers not installed. Please install: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTTING_AVAILABLE = True
except ImportError:
    st.error("Plotting libraries not installed. Please install: pip install matplotlib plotly")
    PLOTTING_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    st.error("Scikit-learn not installed. Please install: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Main title
st.markdown('<div class="main-header">📈 Stock Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["📊 Dashboard", "🔄 Data Processing", "🤖 Model Training", "📈 Predictions", "📋 Results Analysis"]
)

# Check if all required libraries are available
ALL_LIBRARIES_AVAILABLE = all([TF_AVAILABLE, TRANSFORMERS_AVAILABLE, PLOTTING_AVAILABLE, SKLEARN_AVAILABLE])

if not ALL_LIBRARIES_AVAILABLE:
    st.error("Some required libraries are missing. Please install all dependencies.")
    st.stop()

# Helper functions
@st.cache_data
def load_csv_file(file_path):
    """Load CSV file and standardize column names"""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model"""
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not available")
        return None
    try:
        device = -1 if not torch.cuda.is_available() else 0
        return pipeline("sentiment-analysis", 
                       model="distilbert-base-uncased-finetuned-sst-2-english", 
                       device=device)
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None

def get_sentiment(text, sentiment_pipeline):
    """Get sentiment score for text"""
    try:
        result = sentiment_pipeline(text)[0]["label"]
        return 1 if result == "POSITIVE" else (-1 if result == "NEGATIVE" else 0)
    except Exception:
        return 0

def build_lstm_model(input_shape):
    """Build LSTM model"""
    if not TF_AVAILABLE:
        st.error("TensorFlow not available")
        return None
    
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=False),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Welcome to Stock Price Prediction Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Purpose</h3>
            <p>Predict stock prices using LSTM neural networks and sentiment analysis from news data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Features</h3>
            <p>• LSTM Deep Learning Model<br>
            • Sentiment Analysis<br>
            • Technical Indicators<br>
            • Interactive Visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Workflow</h3>
            <p>1. Load Data<br>
            2. Process & Analyze<br>
            3. Train Model<br>
            4. Make Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Quick Start Guide")
    st.markdown("""
    1. **Data Processing**: Upload your CSV files and process the data
    2. **Model Training**: Train the LSTM model on historical stock data
    3. **Predictions**: Generate predictions for new data
    4. **Results Analysis**: View and analyze the results
    """)
    
    # System Information
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        if TF_AVAILABLE and torch.cuda.is_available():
            device = "GPU (CUDA)"
        else:
            device = "CPU"
        st.info(f"🖥️ Computing Device: {device}")
    
    with col2:
        if TF_AVAILABLE:
            tf_version = tf.__version__
            st.info(f"🧠 TensorFlow Version: {tf_version}")
        else:
            st.info("🧠 TensorFlow: Not Available")

# Data Processing Page
elif page == "🔄 Data Processing":
    st.header("Data Processing & Loading")
    
    # File upload section
    st.subheader("📁 Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock_file = st.file_uploader("Upload Stock Data CSV", type=['csv'], key="stock")
        news_file = st.file_uploader("Upload News Data CSV", type=['csv'], key="news")
    
    with col2:
        new_news_file = st.file_uploader("Upload New News Data CSV", type=['csv'], key="new_news")
        actual_file = st.file_uploader("Upload Actual Data CSV", type=['csv'], key="actual")
    
    # Process uploaded files
    if st.button("🔄 Process Data", type="primary"):
        if all([stock_file, news_file, new_news_file, actual_file]):
            try:
                # Load data
                stock_data = pd.read_csv(stock_file)
                news_data = pd.read_csv(news_file)
                new_news_data = pd.read_csv(new_news_file)
                actual_data = pd.read_csv(actual_file)
                
                # Standardize column names
                for df in [stock_data, news_data, new_news_data, actual_data]:
                    df.columns = df.columns.str.strip().str.lower()
                
                # Store in session state
                st.session_state.stock_data = stock_data
                st.session_state.news_data = news_data
                st.session_state.new_news_data = new_news_data
                st.session_state.actual_data = actual_data
                
                st.success("✅ Data loaded successfully!")
                
                # Display data info
                st.subheader("📊 Data Overview")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Stock Data", "News Data", "New News Data", "Actual Data"])
                
                with tab1:
                    st.write(f"**Shape:** {stock_data.shape}")
                    st.write(f"**Columns:** {stock_data.columns.tolist()}")
                    st.dataframe(stock_data.head())
                
                with tab2:
                    st.write(f"**Shape:** {news_data.shape}")
                    st.write(f"**Columns:** {news_data.columns.tolist()}")
                    st.dataframe(news_data.head())
                
                with tab3:
                    st.write(f"**Shape:** {new_news_data.shape}")
                    st.write(f"**Columns:** {new_news_data.columns.tolist()}")
                    st.dataframe(new_news_data.head())
                
                with tab4:
                    st.write(f"**Shape:** {actual_data.shape}")
                    st.write(f"**Columns:** {actual_data.columns.tolist()}")
                    st.dataframe(actual_data.head())
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
        else:
            st.warning("⚠️ Please upload all required CSV files.")

# Model Training Page
elif page == "🤖 Model Training":
    st.header("Model Training")
    
    if 'stock_data' not in st.session_state:
        st.warning("⚠️ Please load data first in the Data Processing section.")
    else:
        stock_data = st.session_state.stock_data.copy()
        news_data = st.session_state.news_data.copy()
        
        # Data preprocessing
        st.subheader("📊 Data Preprocessing")
        
        with st.expander("View Preprocessing Steps"):
            st.write("1. Convert date columns to datetime format")
            st.write("2. Filter stock data from 2019-2024")
            st.write("3. Add technical indicators")
            st.write("4. Perform sentiment analysis on news data")
            st.write("5. Merge stock and news data")
            st.write("6. Scale features")
        
        if st.button("🔄 Start Preprocessing", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Convert dates
                status_text.text("Converting date formats...")
                progress_bar.progress(20)
                
                for df in [stock_data, news_data]:
                    df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
                
                # Filter stock data
                stock_data = stock_data[stock_data["date"] >= "2019-01-01"]
                progress_bar.progress(40)
                
                # Add technical indicators
                status_text.text("Adding technical indicators...")
                stock_data["daily_change"] = stock_data["close"] - stock_data["open"]
                stock_data["volatility"] = stock_data["high"] - stock_data["low"]
                stock_data["50_day_MA"] = stock_data["close"].rolling(window=50).mean()
                stock_data["200_day_MA"] = stock_data["close"].rolling(window=200).mean()
                stock_data.fillna(0, inplace=True)
                progress_bar.progress(60)
                
                # Sentiment analysis
                status_text.text("Performing sentiment analysis...")
                sentiment_pipeline = load_sentiment_model()
                
                if sentiment_pipeline:
                    news_data["sentiment"] = news_data["news_summary"].apply(
                        lambda x: get_sentiment(x, sentiment_pipeline)
                    )
                    news_data = news_data.groupby("date").agg({"sentiment": "mean"}).reset_index()
                    progress_bar.progress(80)
                    
                    # Merge data
                    status_text.text("Merging datasets...")
                    merged_data = pd.merge(stock_data, news_data, on="date", how="left").fillna(0)
                    
                    # Prepare features
                    features = ["sentiment", "volume", "daily_change", "volatility", "50_day_MA", "200_day_MA"]
                    X_train = merged_data[features]
                    y_train = merged_data["close"]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                    
                    st.session_state.X_train = X_train_scaled
                    st.session_state.y_train = y_train
                    st.session_state.scaler = scaler
                    st.session_state.features = features
                    
                    progress_bar.progress(100)
                    status_text.text("Preprocessing completed!")
                    
                    st.success("✅ Data preprocessing completed successfully!")
                    
                    # Display preprocessing results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training Samples", len(X_train_scaled))
                    
                    with col2:
                        st.metric("Features", len(features))
                    
                    with col3:
                        st.metric("Date Range", f"{merged_data['date'].min().strftime('%Y-%m-%d')} to {merged_data['date'].max().strftime('%Y-%m-%d')}")
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
        
        # Model training section
        if 'X_train' in st.session_state:
            st.subheader("🧠 LSTM Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Number of Epochs", 50, 300, 150)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            
            with col2:
                lstm_units = st.slider("LSTM Units", 128, 512, 256)
                dense_units = st.slider("Dense Units", 64, 256, 128)
            
            if st.button("🚀 Train Model", type="primary"):
                try:
                    # Build model
                    model = Sequential([
                        LSTM(lstm_units, input_shape=(1, len(st.session_state.features)), return_sequences=False),
                        Dense(dense_units, activation="relu"),
                        Dense(64, activation="relu"),
                        Dense(1)
                    ])
                    model.compile(loss="mean_squared_error", optimizer="adam")
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Custom callback for progress
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Training... Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f}")
                    
                    # Train model
                    history = model.fit(
                        st.session_state.X_train, 
                        st.session_state.y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[StreamlitCallback()]
                    )
                    
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    st.session_state.training_history = history.history
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display training results
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    fig.update_layout(
                        title="Training Loss Over Time",
                        xaxis_title="Epoch",
                        yaxis_title="Loss"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")

# Predictions Page
elif page == "📈 Predictions":
    st.header("Stock Price Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the Model Training section.")
    else:
        if st.button("🔮 Generate Predictions", type="primary"):
            try:
                new_news_data = st.session_state.new_news_data.copy()
                
                # Convert date
                new_news_data["date"] = pd.to_datetime(new_news_data["date"].astype(str), errors="coerce")
                
                # Sentiment analysis
                sentiment_pipeline = load_sentiment_model()
                if sentiment_pipeline:
                    new_news_data["sentiment"] = new_news_data["news_summary"].apply(
                        lambda x: get_sentiment(x, sentiment_pipeline)
                    )
                    new_news_data = new_news_data.groupby("date").agg({"sentiment": "mean"}).reset_index()
                    
                    # Add last known stock features
                    latest_stock_data = st.session_state.stock_data.iloc[-1]
                    for col in ["volume", "daily_change", "volatility", "50_day_MA", "200_day_MA"]:
                        new_news_data[col] = latest_stock_data[col]
                    
                    # Scale and predict
                    X_test = st.session_state.scaler.transform(new_news_data[st.session_state.features])
                    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                    
                    predicted_prices = st.session_state.model.predict(X_test)
                    new_news_data["predicted_close"] = predicted_prices
                    
                    st.session_state.predictions = new_news_data
                    st.session_state.predictions_made = True
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display predictions
                    st.subheader("📊 Prediction Results")
                    
                    # Create prediction chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=new_news_data["date"],
                        y=new_news_data["predicted_close"],
                        mode='lines+markers',
                        name='Predicted Close Price',
                        line=dict(color='blue', width=2)
                    ))
                    fig.update_layout(
                        title="Predicted Stock Prices",
                        xaxis_title="Date",
                        yaxis_title="Stock Price",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction table
                    st.dataframe(new_news_data[["date", "sentiment", "predicted_close"]])
                    
                    # Download predictions
                    csv = new_news_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"stock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")

# Results Analysis Page
elif page == "📋 Results Analysis":
    st.header("Results Analysis")
    
    if not st.session_state.predictions_made:
        st.warning("⚠️ Please generate predictions first in the Predictions section.")
    else:
        try:
            predictions = st.session_state.predictions
            actual_data = st.session_state.actual_data.copy()
            
            # Convert date in actual data
            actual_data["date"] = pd.to_datetime(actual_data["date"].astype(str), errors="coerce")
            
            # Merge predictions with actual data
            if "close" in actual_data.columns:
                actual_data.rename(columns={"close": "actual_close"}, inplace=True)
                comparison_data = pd.merge(predictions, actual_data, on="date", how="left")
                
                # Filter valid data
                valid_data = comparison_data.dropna(subset=["actual_close", "predicted_close"])
                
                if not valid_data.empty:
                    # Calculate metrics
                    mse = mean_squared_error(valid_data["actual_close"], valid_data["predicted_close"])
                    mae = mean_absolute_error(valid_data["actual_close"], valid_data["predicted_close"])
                    r2 = r2_score(valid_data["actual_close"], valid_data["predicted_close"])
                    
                    # Display metrics
                    st.subheader("📊 Model Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    
                    with col2:
                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                    
                    with col3:
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    # Comparison chart
                    st.subheader("📈 Predicted vs Actual Prices")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=valid_data["date"],
                        y=valid_data["predicted_close"],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=valid_data["date"],
                        y=valid_data["actual_close"],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='red', width=2)
                    ))
                    fig.update_layout(
                        title="Predicted vs Actual Stock Prices",
                        xaxis_title="Date",
                        yaxis_title="Stock Price",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Error analysis
                    st.subheader("📉 Error Analysis")
                    
                    valid_data["error"] = valid_data["actual_close"] - valid_data["predicted_close"]
                    valid_data["error_pct"] = (valid_data["error"] / valid_data["actual_close"]) * 100
                    
                    fig_error = go.Figure()
                    fig_error.add_trace(go.Scatter(
                        x=valid_data["date"],
                        y=valid_data["error"],
                        mode='lines+markers',
                        name='Prediction Error',
                        line=dict(color='orange', width=2)
                    ))
                    fig_error.update_layout(
                        title="Prediction Error Over Time",
                        xaxis_title="Date",
                        yaxis_title="Error (Actual - Predicted)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_error, use_container_width=True)
                    
                    # Statistical summary
                    st.subheader("📋 Statistical Summary")
                    
                    summary_stats = valid_data[["actual_close", "predicted_close", "error", "error_pct"]].describe()
                    st.dataframe(summary_stats)
                    
                    # Download comparison data
                    csv = valid_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison Data",
                        data=csv,
                        file_name=f"prediction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("❌ No valid data available for comparison.")
            else:
                st.error("❌ 'close' column not found in actual data.")
                
        except Exception as e:
            st.error(f"❌ Error in results analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Powered by TensorFlow & Transformers*")
