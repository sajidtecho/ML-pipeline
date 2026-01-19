import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters retrieved from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

class ModelTrainer:
    def __init__(self, model_path='models/'):
        """
        Initialize the ModelTrainer
        
        Args:
            model_path: Directory to save trained models
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def load_data(self, data_path):
        """
        Load processed data for training
        
        Args:
            data_path: Path to the processed data file
            
        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    
    def prepare_data(self, df, target_column, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, **model_params):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_params: Additional parameters for the model
        """
        logger.info("Training the model...")
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating the model...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, model_name='trained_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            model_name: Name of the model file
        """      
        model_file = self.model_path / model_name
        joblib.dump(self.model, model_file)
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_name='trained_model.pkl'):
        """
        Load a trained model from disk
        
        Args:
            model_name: Name of the model file
        """
        model_file = self.model_path / model_name
        self.model = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")


if __name__ == "__main__":
    try:
        params = load_params('params.yaml')
        n_estimators = params['model_training']['n_estimators']
        random_state = params['model_training']['random_state']
        
        trainer = ModelTrainer()
        
        # Load separate train and test data
        train_data = trainer.load_data('./data/processed/train_tfidf.csv')
        test_data = trainer.load_data('./data/processed/test_tfidf.csv')
        
        # Prepare X and y
        target_column = 'label'
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        # Train the model
        trainer.train(X_train, y_train, n_estimators=n_estimators, random_state=random_state)
        
        # Evaluate the model
        metrics = trainer.evaluate(X_test, y_test)
        
        # Save the model
        trainer.save_model('trained_model.pkl')
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")