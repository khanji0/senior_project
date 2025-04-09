# src/components/model_trainer.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                            GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_recall_curve, roc_curve, auc)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, save_metrics_report

# Check for XGBoost availability
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    logging.warning("XGBoost is not available. Will proceed without XGBoost.")

# Create necessary directories
os.makedirs('artifacts/model_report', exist_ok=True)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_path: str = os.path.join('artifacts', 'model_report')
    feature_importance_plot_path: str = os.path.join('artifacts', 'feature_importance.png')
    confusion_matrix_plot_path: str = os.path.join('artifacts', 'confusion_matrix.png')
    roc_curve_plot_path: str = os.path.join('artifacts', 'roc_curve.png')
    pr_curve_plot_path: str = os.path.join('artifacts', 'pr_curve.png')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_report_path, exist_ok=True)

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train and optimize XGBoost model if available
        """
        try:
            if not XGB_AVAILABLE:
                logging.warning("XGBoost is not available. Skipping XGBoost training.")
                return None

            logging.info("Starting XGBoost training")

            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42
            )

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }

            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_xgb_model = grid_search.best_estimator_

            return best_xgb_model

        except Exception as e:
            logging.error(f"Error in XGBoost training: {str(e)}")
            return None

    def plot_feature_importance(self, feature_importance, model_name):
        """
        Plot and save feature importance graph
        """
        try:
            plt.figure(figsize=(12, 6))
            sorted_idx = np.argsort(feature_importance)[::-1]
            pos = np.arange(len(sorted_idx[:10])) + .5

            plt.barh(pos, feature_importance[sorted_idx[:10]])
            plt.yticks(pos, sorted_idx[:10])
            plt.xlabel('Feature Importance Score')
            plt.title(f'{model_name} - Top 10 Feature Importance')

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_trainer_config.model_report_path,
                                    f'{model_name.lower()}_feature_importance.png'))
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def plot_confusion_matrices(self, y_test, predictions_dict):
        """
        Plot confusion matrices for all models
        """
        try:
            n_models = len(predictions_dict)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

            if n_models == 1:
                axes = [axes]

            for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_title(f'{model_name} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')

            plt.tight_layout()
            plt.savefig(self.model_trainer_config.confusion_matrix_plot_path)
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def plot_roc_curves(self, y_test, probas_dict):
        """
        Plot ROC curves for all models
        """
        try:
            plt.figure(figsize=(10, 6))

            for model_name, y_proba in probas_dict.items():
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")

            plt.savefig(self.model_trainer_config.roc_curve_plot_path)
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def plot_model_comparison(self, model_report):
        """
        Plot comparison of model performances
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Extract accuracies
            models = list(model_report.keys())
            train_accuracies = [model_report[model]['train_accuracy'] for model in models]
            test_accuracies = [model_report[model]['test_accuracy'] for model in models]
            
            # Set up bar positions
            x = np.arange(len(models))
            width = 0.35
            
            # Create bars
            plt.bar(x - width/2, train_accuracies, width, label='Train Accuracy')
            plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')
            
            # Customize plot
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            
            # Add value labels on bars
            for i, v in enumerate(train_accuracies):
                plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
            for i, v in enumerate(test_accuracies):
                plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_trainer_config.model_report_path, 
                                    'model_comparison.png'))
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """
        Main method to initiate model training and evaluation
        """
        try:
            logging.info("Starting model training")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define all models with optimized parameters
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight='balanced'
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=2,
                    class_weight='balanced',
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                "AdaBoost": AdaBoostClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=2,
                    class_weight='balanced',
                    random_state=42
                ),
                "K-Nearest Neighbors": KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance'
                ),
                "SVM": SVC(
                    kernel='rbf',
                    C=1.0,
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                )
            }

            # Add XGBoost if available
            if XGB_AVAILABLE:
                xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test)
                if xgb_model is not None:
                    models["XGBoost"] = xgb_model

            # Create voting classifier
            voting_clf = VotingClassifier(
                estimators=[
                    ('rf', models["Random Forest"]),
                    ('gb', models["Gradient Boosting"]),
                    ('ada', models["AdaBoost"])
                ],
                voting='soft'
            )
            models["Voting Classifier"] = voting_clf

            # Train all models
            logging.info("Training all models...")
            predictions_dict = {}
            probas_dict = {}

            for name, model in models.items():
                logging.info(f"Training {name}...")
                if name != "XGBoost":  # XGBoost is already trained if available
                    model.fit(X_train, y_train)
                
                predictions_dict[name] = model.predict(X_test)
                probas_dict[name] = model.predict_proba(X_test)

                # Save feature importance for models that support it
                if hasattr(model, 'feature_importances_'):
                    self.plot_feature_importance(model.feature_importances_, name)

            # Plot evaluation metrics
            self.plot_confusion_matrices(y_test, predictions_dict)
            self.plot_roc_curves(y_test, probas_dict)

            # Print classification reports
            for name, y_pred in predictions_dict.items():
                print(f"\n{name} Classification Report:")
                print(classification_report(y_test, y_pred))

            # Evaluate models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            # Save detailed evaluation report
            save_metrics_report(model_report)

            # Get best model
            best_model_name = max(model_report.keys(),
                                key=(lambda k: model_report[k]['test_accuracy']))
            best_model = model_report[best_model_name]['model']
            best_accuracy = model_report[best_model_name]['test_accuracy']

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Model Accuracy: {best_accuracy:.4f}")

            # Create and save model comparison plot
            self.plot_model_comparison(model_report)

            return best_model_name, best_accuracy, model_report

        except Exception as e:
            raise CustomException(e, sys)