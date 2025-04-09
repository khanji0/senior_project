# src/pipeline/train_pipeline.py

import sys
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")

            # Data Ingestion
            logging.info("Data Ingestion started")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed")

            # Data Transformation
            logging.info("Data Transformation started")
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_path,
                test_path
            )
            logging.info("Data Transformation completed")

            # Model Training
            logging.info("Model Training started")
            best_model_name, best_accuracy, model_report = self.model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )
            logging.info("Model Training completed")

            # Print detailed results
            print("\n" + "="*50)
            print("Training Results:")
            print("="*50)
            print(f"Best Model: {best_model_name}")
            print(f"Best Model Accuracy: {best_accuracy:.4f}")
            
            # Print performance of all models
            print("\nAll Models Performance:")
            print("-"*50)
            for model_name, metrics in model_report.items():
                print(f"\n{model_name}:")
                print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
                print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
                if 'f1_score' in metrics:
                    print(f"F1 Score: {metrics['f1_score']:.4f}")
            print("="*50)

            return {
                "best_model_name": best_model_name,
                "best_accuracy": best_accuracy,
                "model_report": model_report
            }

        except Exception as e:
            raise CustomException(e, sys)