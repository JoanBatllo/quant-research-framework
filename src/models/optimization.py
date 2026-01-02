import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

def optimize_hyperparameters(
    data: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str = 'Target', 
    model_type: str = 'rf', 
    n_trials: int = 20
) -> dict:
    """
    Finds the best hyperparameters for a given model type using Optuna.
    
    Uses TimeSeriesSplit for cross-validation within the optimization process
    to avoid look-ahead bias during tuning.
    """
    
    # Prepare X and y
    X = data[feature_cols]
    y = data[target_col]
    
    # Use TimeSeriesSplit explicitly for validation during tuning
    # This splits the data into K sequential splits
    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective(trial):
        params = {}
        model = None
        
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'eval_metric': 'logloss',
                'n_jobs': -1
            }
            model = XGBClassifier(**params)
        
        # Cross-Validation Loop (Manual to support TimeSeriesSplit easily)
        accuracies = []
        
        for train_index, val_index in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
            y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
            
            model.fit(X_train_cv, y_train_cv)
            preds = model.predict(X_val_cv)
            acc = accuracy_score(y_val_cv, preds)
            accuracies.append(acc)
            
        # Return average accuracy across folds
        return sum(accuracies) / len(accuracies)

    # Create Study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f" Optimization finished. Best params: {study.best_params}")
    print(f" Best validation accuracy: {study.best_value:.2%}")
    
    return study.best_params
