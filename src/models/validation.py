
import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
from sklearn.metrics import accuracy_score, classification_report

def walk_forward_validation(data: pd.DataFrame, 
                            feature_cols: list[str], 
                            target_col: str = 'Target', 
                            train_window_years: int = 5, 
                            test_window_years: int = 1,
                            model_type: str = 'rf') -> dict:
    """
    Performs Walk-Forward Validation (Rolling Window Analysis).
    
    Simulates a realistic trading scenario where the model is periodically re-trained
    using the most recent data available.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with DatetimeIndex.
    feature_cols : list[str]
        Features to use.
    target_col : str
        Name of target column.
    train_window_years : int
        Number of years of history to use for training.
    test_window_years : int
        Number of years to predict forward (and re-train frequency).
    model_type : str
        'rf' or 'xgb'.

    Returns
    -------
    dict
        Results containing:
        - 'predictions': pd.Series (concatenated out-of-sample predictions)
        - 'metrics': list of dicts (performance per window)
        - 'overall_accuracy': float
    """
    
    # Sort data just in case
    data = data.sort_index()
    
    start_date = data.index.min()
    end_date = data.index.max()
    
    # We need at least (train_window) years of data to start
    current_train_start = start_date
    current_test_start = start_date + pd.DateOffset(years=train_window_years)
    
    all_predictions = []
    window_metrics = []
    
    print(f"Starting Walk-Forward Validation ({model_type})...")
    
    while current_test_start < end_date:
        # Define window boundaries
        current_test_end = current_test_start + pd.DateOffset(years=test_window_years)
        
        # 1. Slice Data
        # Train: [Start, Start + 5 years]
        train_data = data[(data.index >= current_train_start) & (data.index < current_test_start)]
        
        # Test: [Start + 5 years, Start + 6 years]
        test_data = data[(data.index >= current_test_start) & (data.index < current_test_end)]
        
        if len(test_data) == 0:
            break
            
        # 2. Train Model
        trainer = ModelTrainer(data, feature_cols, target=target_col)
        # We manually overwrite internal train/test sets to match our rolling window
        trainer.X_train = train_data[feature_cols]
        trainer.y_train = train_data[target_col]
        
        # Note: We don't really use trainer.split_data() here because we control the split
        trainer.model = trainer.train(model_type=model_type)
        
        # 3. Predict Out-of-Sample (Test)
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        preds = trainer.model.predict(X_test)
        
        # Save results
        window_acc = accuracy_score(y_test, preds)
        window_metrics.append({
            "period": f"{current_test_start.date()} to {current_test_end.date()}",
            "accuracy": window_acc,
            "train_samples": len(train_data),
            "test_samples": len(test_data)
        })
        
        # Create a Series for these predictions
        pred_series = pd.Series(preds, index=test_data.index)
        all_predictions.append(pred_series)
        
        # 4. Move Window Forward
        # Rolling Window: Move both start and end forward
        # (Alternatively: Expanding Window would keep start_date fixed)
        current_train_start += pd.DateOffset(years=test_window_years)
        current_test_start += pd.DateOffset(years=test_window_years)
        
    # Concatenate all test predictions into one seamless series
    if all_predictions:
        full_pred_series = pd.concat(all_predictions)
        
        # Calculate global accuracy on the concatenated results
        # We need to match the indices from the original data
        aligned_targets = data.loc[full_pred_series.index][target_col]
        overall_acc = accuracy_score(aligned_targets, full_pred_series)
    else:
        full_pred_series = pd.Series()
        overall_acc = 0.0

    return {
        "predictions": full_pred_series,
        "metrics": window_metrics,
        "overall_accuracy": overall_acc
    }
