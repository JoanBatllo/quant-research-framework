import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

def run_feature_selection(
    data: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str = 'Target', 
    model_type: str = 'rf'
) -> dict:
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to match
    the optimal subset of features.
    
    Args:
        data: Global Dataframe with features and target.
        feature_cols: List of candidate feature names.
        target_col: Name of target column.
        model_type: 'rf' or 'xgb'.
        
    Returns:
        dict: {
            'selected_features': list[str],
            'dropped_features': list[str],
            'n_features_in': int,
            'n_features_out': int
        }
    """
    X = data[feature_cols]
    y = data[target_col]
    
    # 1. Define Estimator
    if model_type == 'rf':
        estimator = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_type == 'xgb':
        estimator = XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.05, 
            random_state=42, 
            eval_metric='logloss',
            n_jobs=-1
        )
    else:
        raise ValueError("Invalid model type")

    # 2. Define CV Strategy (TimeSeriesSplit for finance)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 3. Run RFECV
    # step=1: remove 1 feature at a time
    # min_features_to_select=3: ensure we keep at least 3 basic features
    selector = RFECV(estimator, step=1, cv=tscv, scoring='accuracy', min_features_to_select=3, n_jobs=-1)
    
    print("Running RFECV... this may take a moment.")
    selector.fit(X, y)
    
    # 4. Process Results
    selected_mask = selector.support_
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
    dropped_features = [f for f, s in zip(feature_cols, selected_mask) if not s]
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'n_features_in': len(feature_cols),
        'n_features_out': len(selected_features),
        'grid_scores': selector.cv_results_['mean_test_score'].tolist() if hasattr(selector, 'cv_results_') else []
    }
