import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data.loaders import get_price_data
from src.data.preprocessing import prepare_price_data, standardize_price_dataframe, add_return_columns
from src.data.utils import load_multi_asset_data
from src.backtesting.strategies import always_long, moving_average_crossover, ml_strategy
from src.backtesting.engine import run_backtest
from src.analysis.performance import summarize_performance
from src.data.features import generate_features
from src.models.trainer import ModelTrainer

st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

st.title("Quant Research Framework")
st.markdown("Professional quantitative analysis and machine learning platform.")

# Sidebar - Global Settings
st.sidebar.header("Configuration")
TICKER_OPTIONS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "BTC-USD", "ETH-USD", "GLD", "TLT", "EEM"]
ticker = st.sidebar.selectbox("Target Ticker (Analysis & Backtest)", options=TICKER_OPTIONS, index=0)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Tabs
tab_backtest, tab_ml, tab_portfolio = st.tabs(["Strategy Backtester", "ML Laboratory", "Portfolio Manager"])

# --- TAB 1: BACKTESTER ---
with tab_backtest:
    st.markdown("### Classic Strategy Backtesting")
    
    col_strat, col_act = st.columns([2, 1])
    
    with col_strat:
        # Dynamic Strategy List
        strat_options = ["Always Long (Buy & Hold)", "Simple MA Crossover"]
        if 'ml_model' in st.session_state:
            strat_options.append("ML Strategy (Trained Model)")
            
        strategy_name = st.selectbox("Select Strategy", strat_options)
    
    # Strat Params
    fast_window, slow_window = 10, 50
    if strategy_name == "Simple MA Crossover":
        col_p1, col_p2 = st.columns(2)
        fast_window = col_p1.slider("Fast MA", 2, 50, 10)
        slow_window = col_p2.slider("Slow MA", 10, 200, 50)
    elif strategy_name == "ML Strategy (Trained Model)":
        st.caption(f"Using trained model with features: {st.session_state['ml_features']}")
        
        # IN-SAMPLE WARNING
        if 'train_split_date' in st.session_state:
            train_date = pd.to_datetime(st.session_state['train_split_date']).date()
            if start_date < train_date:
                st.warning(f"⚠️ DANGER: You are backtesting on Training Data (Before {train_date}). Results will be unrealistic (Overfitting).")

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
                elif strategy_name == "ML Strategy (Trained Model)":
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
    st.markdown("### Machine Learning Model Training")
    st.info("Train a Global Model on multiple assets to learn universal market patterns.")
    
    col_ml_1, col_ml_2, col_ml_3 = st.columns(3)
    split_date = col_ml_1.date_input("Train/Test Split Date", pd.to_datetime("2023-01-01"))
    model_choice = col_ml_2.selectbox("Model Type", ["Random Forest", "XGBoost"])
    
    # Internal code mapping
    model_map = {"Random Forest": "rf", "XGBoost": "xgb"}
    selected_model_code = model_map[model_choice]
    
    # Multi-Asset Selector
    training_tickers = st.multiselect("Training Universe (Global Data)", TICKER_OPTIONS, default=["SPY", "QQQ", "IWM", "GLD", "BTC-USD"])

    # --- HYPERPARAMETER OPTIMIZATION ---
    with st.expander("Optimize Hyperparameters (Optuna)", expanded=False):
        n_trials = st.number_input("Number of Trials", min_value=5, max_value=100, value=20)
        
        if st.button("Run Global Optimization"):
            with st.spinner(f"Optimizing {model_choice} on {len(training_tickers)} assets..."):
                try:
                    from src.models.optimization import optimize_hyperparameters
                    
                    # 1. Load Global Data
                    df_opt_global = load_multi_asset_data(training_tickers, "2010-01-01", str(end_date))
                    
                    # Drop Ticker col for training
                    optimize_cols = [c for c in df_opt_global.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Target', 'Ticker']]
                    
                    # 2. Run Optimization using TimeSeriesSplit validation
                    best_params = optimize_hyperparameters(
                        df_opt_global, 
                        optimize_cols, 
                        model_type=selected_model_code, 
                        n_trials=n_trials
                    )
                    
                    # 3. Save to Session State
                    st.session_state['best_params'] = best_params
                    st.session_state['opt_model_type'] = selected_model_code
                    
                    st.success("Optimization Complete! Best parameters saved.")
                    st.json(best_params)
                    
                except Exception as e:
                    st.error(f"Optimization Error: {e}")

    # --- FEATURE SELECTION (RFE) ---
    with st.expander("Feature Optimization (RFE)", expanded=False):
        st.info("Recursively remove noisy features to improve model stability.")
        if st.button("Run Feature Elimination"):
            with st.spinner(f"Running RFECV on Global Data..."):
                try:
                    from src.models.selection import run_feature_selection
                    
                    # 1. Load Data (if not already loaded logic, but for simplicity we reload)
                    df_rfe = load_multi_asset_data(training_tickers, "2010-01-01", str(end_date))
                    
                    # Feature candidates
                    cand_cols = [c for c in df_rfe.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Target', 'Ticker']]
                    
                    # 2. Run RFE
                    rfe_res = run_feature_selection(df_rfe, cand_cols, model_type=selected_model_code)
                    
                    # 3. Save
                    st.session_state['selected_features'] = rfe_res['selected_features']
                    
                    st.success(f"Optimization Complete! Kept {rfe_res['n_features_out']} features (Dropped {len(rfe_res['dropped_features'])}).")
                    
                    col_rfe1, col_rfe2 = st.columns(2)
                    col_rfe1.write("**Kept Features:**")
                    col_rfe1.code(rfe_res['selected_features'])
                    
                    col_rfe2.write("**Dropped (Noise):**")
                    col_rfe2.code(rfe_res['dropped_features'])
                    
                except Exception as e:
                    st.error(f"RFE Error: {e}")

    # Check if we have optimized params for the selected model
    use_optimized = False
    if 'best_params' in st.session_state and st.session_state.get('opt_model_type') == selected_model_code:
        st.success(f"Running with Optimized Parameters found for {model_choice}!")
        use_optimized = True
    
    if st.button("Train Global Model", key="btn_train"):
        with st.spinner(f"Training on {len(training_tickers)} assets..."):
            try:
                # 1. Load Multi-Asset Data
                df_ml = load_multi_asset_data(training_tickers, "2010-01-01", str(end_date)) 

                # 2. Features are already generated inside load_multi_asset_data
                df_features = df_ml
                
                # 3. Setup Trainer (exclude Ticker and raw cols)
                # 3. Setup Trainer (exclude Ticker and raw cols)
                # Check if we have a SELECTED subset of features from RFE
                if 'selected_features' in st.session_state:
                    feature_cols = st.session_state['selected_features']
                    st.info(f"Training using optimized subset of {len(feature_cols)} features.")
                else:
                    feature_cols = [c for c in df_features.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Target', 'Ticker']]
                
                
                trainer = ModelTrainer(df_features, feature_cols, target='Target')
                
                # 4. Split and Train
                trainer.split_data(str(split_date))
                
                # TRAIN WITH OR WITHOUT OPTIMIZED PARAMS
                if use_optimized:
                    model = trainer.train(model_type=selected_model_code, **st.session_state['best_params'])
                else:
                    model = trainer.train(model_type=selected_model_code)
                
                # Save to Session State
                st.session_state['ml_model'] = model
                st.session_state['ml_features'] = feature_cols
                st.session_state['df_features'] = df_features # This is now the Global Dataset
                st.session_state['model_code'] = selected_model_code 
                st.session_state['train_split_date'] = split_date # Save split date for backtest warning
                
                st.success(f"Global Model Trained successfully on {len(training_tickers)} assets!")

            except Exception as e:
                st.error(f"ML Error: {e}")

    # --- PERSISTENT RESULTS SECTION ---
    if 'ml_model' in st.session_state and 'df_features' in st.session_state:
        st.markdown("---")
        
        # We reconstruct a temporary trainer to use its plotting helpers
        df_feats = st.session_state['df_features']
        feats = st.session_state['ml_features']
        # We need to re-split to get X_test for evaluation
        trainer_viz = ModelTrainer(df_feats, feats, target='Target')
        trainer_viz.split_data(str(split_date))
        trainer_viz.model = st.session_state['ml_model'] # Attach trained model
        
        # Evaluate
        acc, report = trainer_viz.evaluate()
        
        col_metrics_1, col_metrics_2 = st.columns(2)
        col_metrics_1.metric("Global Test Accuracy", f"{acc:.2%}")
        col_metrics_2.text(f"Model: {st.session_state.get('model_code', 'Unknown')}")

        st.subheader("Model Insights")
        
        tab_viz_1, tab_viz_2, tab_viz_3 = st.tabs(["Feature Importance", "Confusion Matrix", "Walk-Forward Validation"])
        
        with tab_viz_1:
            fig_imp = trainer_viz.plot_feature_importance()
            st.pyplot(fig_imp)
            
        with tab_viz_2:
            fig_cm = trainer_viz.plot_confusion_matrix()
            st.pyplot(fig_cm)
            st.text("Classification Report:")
            st.dataframe(pd.DataFrame(report).transpose())

        with tab_viz_3:
            st.markdown("### Robustness Check: Walk-Forward Validation")
            st.info("This simulates a realistic scenario where the model is re-trained every year.")
            
            col_wf_1, col_wf_2 = st.columns(2)
            train_window = col_wf_1.number_input("Train Window (Years)", min_value=1, value=5)
            test_window = col_wf_2.number_input("Test Window (Years)", min_value=1, value=1)
            
            if st.button("Run Walk-Forward Analysis", key="btn_wf"):
                with st.spinner("Running Rolling Window Analysis (This might take a while on Global Data)..."):
                    from src.models.validation import walk_forward_validation

                    results = walk_forward_validation(
                        data=df_feats,
                        feature_cols=feats,
                        target_col='Target',
                        train_window_years=train_window,
                        test_window_years=test_window,
                        model_type=st.session_state.get('model_code', 'rf')
                    )
                    
                    st.success(f"Analysis Complete!")
                    st.metric("Overall Out-of-Sample Accuracy", f"{results['overall_accuracy']:.2%}")
                    
                    res_df = pd.DataFrame(results['metrics'])
                    if not res_df.empty:
                        st.dataframe(res_df.style.highlight_max(axis=0, subset=['accuracy'], color='lightgreen'))
                        st.subheader("Accuracy per Period")
                        st.bar_chart(res_df.set_index("period")['accuracy'])
                    else:
                        st.warning("Not enough data.")

# --- TAB 3: PORTFOLIO MANAGER ---
with tab_portfolio:
    st.markdown("### Global Portfolio Manager")
    st.info("Use your Global Model to actively manage a portfolio of assets.")
    
    if 'ml_model' not in st.session_state:
        st.warning("⚠️ Please train a Global Model in 'ML Laboratory' first!")
    else:
        col_pm_1, col_pm_2, col_pm_3 = st.columns(3)
        
        # Portfolio Config
        pf_tickers = col_pm_1.multiselect("Portfolio Universe", TICKER_OPTIONS, default=["SPY", "QQQ", "IWM", "GLD", "TLT"])
        top_n = col_pm_2.slider("Select Top N Assets to Hold", 1, 5, 2)
        cost_bps = col_pm_3.number_input("Transaction Cost (bps)", value=10)
        
        pf_start_date = st.date_input("Backtest Start Date", pd.to_datetime("2023-01-01"))
        
        # IN-SAMPLE WARNING (Portfolio)
        if 'train_split_date' in st.session_state:
            train_date = pd.to_datetime(st.session_state['train_split_date']).date()
            if pf_start_date < train_date:
                st.warning(f"⚠️ DANGER: You are simulating on Training Data (Before {train_date}). Results are overfitted and unrealistic.")
        
        if st.button("🚀 Run Portfolio Backtest", key="btn_pf_run"):
            with st.spinner(f"Simulating Portfolio Strategy on {len(pf_tickers)} assets..."):
                try:
                    from src.backtesting.portfolio import VectorizedBacktester
                    
                    # 1. Load Data
                    df_global = load_multi_asset_data(pf_tickers, "2010-01-01", str(end_date))
                    
                    # Filter for backtest period onwards (for efficiency, but we need some history for features)
                    # Actually, features are already generated. We just need to make predictions.
                    
                    # 2. Predict Probabilities
                    feature_cols = st.session_state.get('ml_features', [])
                    if not feature_cols:
                         # Fallback if not saved explicitly
                         feature_cols = [c for c in df_global.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Target', 'Ticker']]

                    # Predict
                    X_pred = df_global[feature_cols]
                    model = st.session_state['ml_model']
                    probs = model.predict_proba(X_pred)[:, 1] # Probability of Class 1 (Up)
                    
                    df_global['Probability'] = probs
                    
                    # 3. Pivot to Matrix form (Date x Ticker)
                    # We need 'Return' and 'Probability' matrices
                    probs_matrix = df_global.pivot_table(index=df_global.index, columns='Ticker', values='Probability')
                    returns_matrix = df_global.pivot_table(index=df_global.index, columns='Ticker', values='Return')
                    
                    # 4. Filter Date Range
                    probs_matrix = probs_matrix[str(pf_start_date):]
                    returns_matrix = returns_matrix[str(pf_start_date):]
                    
                    # 5. Run Backtest
                    vbt = VectorizedBacktester(probs_matrix, returns_matrix)
                    pf_results = vbt.run_top_n_strategy(n=top_n, transaction_cost_bps=cost_bps)
                    
                    # 6. Benchmark (Equal Weight Buy & Hold of Universe)
                    # Simple average of returns of all assets in universe
                    bench_ret = returns_matrix.mean(axis=1)
                    bench_equity = (1 + bench_ret).cumprod()
                    bench_equity = bench_equity / bench_equity.iloc[0]
                    
                    # 7. Visualization
                    st.success("Backtest Complete!")
                    
                    # Metrics
                    total_ret = pf_results['Equity'].iloc[-1] - 1
                    bench_total_ret = bench_equity.iloc[-1] - 1
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Portfolio Return", f"{total_ret:.2%}", delta=f"{(total_ret - bench_total_ret):.2%}")
                    m2.metric("Benchmark Return", f"{bench_total_ret:.2%}")
                    m3.metric("Avg Daily Turnover", f"{pf_results['Turnover'].mean():.2%}")
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pf_results.index, y=pf_results['Equity'], name=f"Top {top_n} Strategy", line=dict(color='#00CC96', width=2)))
                    fig.add_trace(go.Scatter(x=bench_equity.index, y=bench_equity, name="Equal Weight Universe", line=dict(color='#636EFA', dash='dot')))
                    fig.update_layout(template="plotly_dark", title="Portfolio Equity Curve vs Benchmark", xaxis_title="Date", yaxis_title="Equity (Normalized)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Weights Visualization (Area Chart)
                    st.subheader("Asset Allocation Over Time")
                    # We need to re-calculate weights to visualize them or modify VBT to return them.
                    # For now, let's just show top holdings recently?
                    # Too complex to re-calc here quickly. Let's just stick to performance.
                    
                except Exception as e:
                    st.error(f"Portfolio Backtest Error: {e}")
