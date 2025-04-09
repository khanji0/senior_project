# src/utils.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def create_model_evaluation_plots(y_true, y_pred, y_pred_proba, model_name):
    try:
        # Create directory for plots if it doesn't exist
        os.makedirs('artifacts/plots', exist_ok=True)

        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'artifacts/plots/{model_name}_confusion_matrix.png')
        plt.close()

        # 2. ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'artifacts/plots/{model_name}_roc_curve.png')
        plt.close()

        # 3. Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.savefig(f'artifacts/plots/{model_name}_precision_recall_curve.png')
        plt.close()

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Create progress bar
        pbar = tqdm(models.items(), desc="Training and evaluating models")

        for model_name, model in pbar:
            pbar.set_description(f"Processing {model_name}")

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Get probability predictions if available
            if hasattr(model, "predict_proba"):
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_pred_proba = y_test_pred

            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)

            # Generate classification report
            clf_report = classification_report(y_test, y_test_pred)

            # Create evaluation plots
            create_model_evaluation_plots(y_test, y_test_pred, y_test_pred_proba, model_name)

            # Store all metrics
            report[model_name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'classification_report': clf_report,
                'test_predictions': y_test_pred,
                'test_probabilities': y_test_pred_proba
            }

            # Log results
            logging.info(f"\n{model_name} Results:")
            logging.info(f"Training Time: {training_time:.2f} seconds")
            logging.info(f"Train Accuracy: {train_accuracy:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info("\nClassification Report:")
            logging.info(f"\n{clf_report}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def save_metrics_report(model_report, output_path='artifacts/model_evaluation_report.txt'):
    try:
        with open(output_path, 'w') as f:
            f.write("Model Evaluation Report\n")
            f.write("=====================\n\n")

            # Sort models by test accuracy
            sorted_models = sorted(model_report.items(),
                                 key=lambda x: x[1]['test_accuracy'],
                                 reverse=True)

            for model_name, metrics in sorted_models:
                f.write(f"\n{model_name}\n")
                f.write("=" * len(model_name) + "\n")
                f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n")
                f.write(f"Train Accuracy: {metrics['train_accuracy']:.4f}\n")
                f.write(f"Test Accuracy: {metrics['test_accuracy']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(f"{metrics['classification_report']}\n")
                f.write("-" * 50 + "\n")

    except Exception as e:
        raise CustomException(e, sys)