import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ModelTrainer:
    def __init__(self, data, features, target='Target'):
        """
        Initialize the trainer.
        :param data: DataFrame containing features and target.
        :param features: List of column names to be used as features.
        :param target: Name of the target column.
        """
        self.data = data
        self.features = features
        self.target = target
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
    
    def split_data(self, split_date='2023-01-01'):
        """
        Split data into train and test sets based on a date.
        """
        train = self.data[self.data.index < split_date].copy()
        test = self.data[self.data.index >= split_date].copy()
        
        self.X_train = train[self.features]
        self.y_train = train[self.target]
        self.X_test = test[self.features]
        self.y_test = test[self.target]
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, n_estimators=100, max_depth=5, random_state=42):
        """
        Train a Random Forest Classifier.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance if any
        )
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """
        Make predictions and return accuracy and report.
        """
        if not self.model:
            raise ValueError("Model not trained yet!")
            
        self.predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.predictions)
        report = classification_report(self.y_test, self.predictions, output_dict=True)
        
        return accuracy, report
        
    def plot_feature_importance(self):
        """
        Plot feature importance.
        """
        if not self.model:
            return None
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(self.features)[indices], ax=ax, palette="viridis")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        return fig

    def plot_confusion_matrix(self):
        """
        Plot confusion matrix.
        """
        if self.predictions is None:
            return None
            
        cm = confusion_matrix(self.y_test, self.predictions)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title("Confusion Matrix")
        return fig
