import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data.loaders import get_price_data
from src.data.preprocessing import prepare_price_data, standardize_price_dataframe, add_return_columns
from src.backtesting.strategies import always_long, moving_average_crossover, ml_strategy
from src.backtesting.engine import run_backtest
from src.analysis.performance import summarize_performance
from src.data.features import generate_features
from src.models.trainer import ModelTrainer

st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

st.title("Quant Research Framework Dashboard")

# Sidebar - Global Settings
st.sidebar.header("Global Settings")
TICKER_OPTIONS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "BTC-USD", "ETH-USD"]
ticker = st.sidebar.selectbox("Ticker Symbol", options=TICKER_OPTIONS, index=0)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Tabs
tab_backtest, tab_ml = st.tabs(["📈 Strategy Backtester", "🤖 ML Laboratory"])

# --- TAB 1: BACKTESTER ---
with tab_backtest:
    st.markdown("### Classic Strategy Backtesting")
    
    col_strat, col_act = st.columns([2, 1])
    
    with col_strat:
        # Dynamic Strategy List
        strat_options = ["Always Long (Buy & Hold)", "Simple MA Crossover"]
        if 'ml_model' in st.session_state:
            strat_options.append("🤖 ML Strategy (Trained Model)")
            
        strategy_name = st.selectbox("Select Strategy", strat_options)
    
    # Strat Params
    fast_window, slow_window = 10, 50
    if strategy_name == "Simple MA Crossover":
        col_p1, col_p2 = st.columns(2)
        fast_window = col_p1.slider("Fast MA", 2, 50, 10)
        slow_window = col_p2.slider("Slow MA", 10, 200, 50)
    elif strategy_name == "🤖 ML Strategy (Trained Model)":
        st.caption(f"Using trained model with features: {st.session_state['ml_features']}")

    with col_act:
        st.write("") # Spacer
        st.write("")
        run_btn = st.button("Run Backtest", key="btn_run")
        compare_btn = st.button("Compare with Benchmark", key="btn_compare")

    if run_btn or compare_btn:
        with st.spinner("Running Backtest..."):
            try:
                # Load & Preprocess
                df = get_price_data(ticker, str(start_date), str(end_date))
                df = standardize_price_dataframe(df)
                df = add_return_columns(df)

                # Define Logic
                if strategy_name == "Always Long (Buy & Hold)":
                    signal = always_long(df)
                    expl = "Buys and holds the asset."
                elif strategy_name == "Simple MA Crossover":
                    signal = moving_average_crossover(df, fast_window, slow_window)
                    expl = f"Buy when MA({fast_window}) > MA({slow_window})."
                elif strategy_name == "🤖 ML Strategy (Trained Model)":
                    # Generate features on the fly for the backtest period
                    df = generate_features(df)
                    signal = ml_strategy(df, st.session_state['ml_model'], st.session_state['ml_features'])
                    expl = "Uses the Random Forest model trained in the ML Lab."

                res = run_backtest(df, signal)
                metrics = summarize_performance(res.equity_curve)

                if compare_btn:
                    # Run Benchmark
                    sig_bench = always_long(df)
                    res_bench = run_backtest(df, sig_bench)
                    met_bench = summarize_performance(res_bench.equity_curve)
                    
                    st.subheader("Comparison Results")
                    
                    # Metrics Table
                    comp_df = pd.DataFrame({
                        "Metric": metrics.keys(),
                        "Selected Strategy": metrics.values(),
                        "Benchmark (Buy&Hold)": met_bench.values()
                    }).set_index("Metric")
                    st.table(comp_df)
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res.equity_curve.index, y=res.equity_curve, name="Strategy"))
                    fig.add_trace(go.Scatter(x=res_bench.equity_curve.index, y=res_bench.equity_curve, name="Benchmark"))
                    fig.update_layout(template="plotly_dark", title="Equity Curve Comparison")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # Single Run
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
                    col2.metric("Sharpe", f"{metrics['Sharpe Ratio']:.2f}")
                    col3.metric("Volatility", f"{metrics['Volatility (%)']:.2f}%")
                    col4.metric("Max DD", f"{metrics['Max Drawdown (%)']:.2f}%")
                    
                    st.line_chart(res.equity_curve)

            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: ML LABORATORY ---
with tab_ml:
    st.markdown("### 🧬 Machine Learning Model Training")
    st.info("Here we train a Random Forest model to predict if tomorrow's return will be positive based on technical features.")
    
    col_ml_1, col_ml_2, col_ml_3 = st.columns(3)
    split_date = col_ml_1.date_input("Train/Test Split Date", pd.to_datetime("2023-01-01"))
    model_choice = col_ml_2.selectbox("Model Type", ["Random Forest", "XGBoost"])
    
    if st.button("🚀 Train Model", key="btn_train"):
        with st.spinner("Generating Features & Training Model..."):
            try:
                # 1. Load Data
                df_ml = get_price_data(ticker, "2010-01-01", str(end_date)) # Use long history for ML
                
                # Preprocess (Standardize + Returns)
                df_ml = standardize_price_dataframe(df_ml)
                # Note: generate_features already calculates returns if missing, but we call it here for consistency
                # df_ml = add_return_columns(df_ml) 

                # 2. Features
                df_features = generate_features(df_ml)
                st.write(f"Data Shape after Feature Engineering: {df_features.shape}")
                
                # 3. Setup Trainer
                feature_cols = [c for c in df_features.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Target']]
                trainer = ModelTrainer(df_features, feature_cols, target='Target')
                
                # 4. Split
                X_train, X_test, y_train, y_test = trainer.split_data(str(split_date))
                
                col_tr, col_te = st.columns(2)
                col_tr.metric("Training Samples", len(X_train))
                col_te.metric("Test Samples", len(X_test))
                
                # 5. Train
                # Map standard names to internal codes
                model_map = {"Random Forest": "rf", "XGBoost": "xgb"}
                selected_model_code = model_map[model_choice]
                
                model = trainer.train(model_type=selected_model_code)
                
                # Save to Session State
                st.session_state['ml_model'] = model
                st.session_state['ml_features'] = feature_cols
                
                # 6. Evaluate
                acc, report = trainer.evaluate()
                
                st.success(f"Model Trained! Test Accuracy: **{acc:.2%}**. Saved to Session State!")
                
                # 7. Visualizations
                st.subheader("Model Insights")
                
                # Feature Importance
                fig_imp = trainer.plot_feature_importance()
                st.pyplot(fig_imp)
                
                # Confusion Matrix
                fig_cm = trainer.plot_confusion_matrix()
                st.pyplot(fig_cm)
                
                # Detailed Report
                st.subheader("Classification Report")
                st.text(pd.DataFrame(report).transpose())
                
                # --- WALK-FORWARD VALIDATION SECTION ---
                st.markdown("---")
                st.markdown("### 🧪 Robustness Check: Walk-Forward Validation")
                st.info("This simulates a realistic scenario where the model is re-trained every year. It prevents 'cheating' by strictly separating past and future data.")
                
                with st.expander("Run Walk-Forward Analysis (Advanced)", expanded=False):
                    col_wf_1, col_wf_2 = st.columns(2)
                    train_window = col_wf_1.number_input("Train Window (Years)", min_value=1, value=5)
                    test_window = col_wf_2.number_input("Test Window (Years)", min_value=1, value=1)
                    
                    if st.button("🏃‍♂️ Run Walk-Forward Test"):
                        with st.spinner("Running Rolling Window Analysis... This may take a minute..."):
                            # Get data again just to be safe (or reuse df_features)
                            from src.models.validation import walk_forward_validation
                            
                            # Map model choice again
                            model_map = {"Random Forest": "rf", "XGBoost": "xgb"}
                            selected_model_code = model_map[model_choice]
                            
                            results = walk_forward_validation(
                                data=df_features,
                                feature_cols=feature_cols,
                                target_col='Target',
                                train_window_years=train_window,
                                test_window_years=test_window,
                                model_type=selected_model_code
                            )
                            
                            # Display Results
                            st.success(f"Analysis Complete!")
                            st.metric("🏆 Overall Realistic Accuracy", f"{results['overall_accuracy']:.2%}")
                            
                            # Metrics DataFrame
                            res_df = pd.DataFrame(results['metrics'])
                            if not res_df.empty:
                                st.dataframe(res_df.style.highlight_max(axis=0, subset=['accuracy'], color='lightgreen'))
                            
                                # Plot Accuracy over Time
                                st.subheader("Accuracy per Period")
                                st.bar_chart(res_df.set_index("period")['accuracy'])
                            else:
                                st.warning("Not enough data to run full validation window.")

            except Exception as e:
                st.error(f"ML Error: {e}")

