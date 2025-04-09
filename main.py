# main.py

from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException
from src.logger import logging
import sys

def main():
    try:
        pipeline = TrainPipeline()
        results = pipeline.run_pipeline()

        print("\n" + "="*50)
        print("Training Pipeline Completed Successfully!")
        print("="*50)
        print(f"Best Model: {results['best_model_name']}")
        print(f"Best Model Accuracy: {results['best_accuracy']:.4f}")

        # Additional detailed results are already printed in the pipeline
        print("="*50 + "\n")

    except Exception as e:
        logging.error("An error occurred in the main function")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()

