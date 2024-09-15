import pandas as pd
import numpy as np
import joblib  # For saving the trained model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class TrainPipeline:
    def __init__(self):
        # You can add parameters here or load configurations from a file if needed
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_path = 'models/trained_model.pkl'  # Path to save the model

    def load_data(self, filepath):
        """
        Load dataset from a CSV file.
        """
        data = pd.read_csv('D:\\MLProject@2003\\Notebook\\stud.csv')
        return data

    def preprocess_data(self, data):
        """
        Preprocess the dataset: handle missing values, encode categorical variables,
        and scale numerical features.
        """
        # Example preprocessing (to be customized according to your dataset)
        data = pd.get_dummies(data, drop_first=True)  # One-hot encoding for categorical variables
        X = data.drop('target_column', axis=1)  # Replace 'target_column' with your target column name
        y = data['target_column']
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numerical features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train the machine learning model.
        """
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set and print metrics.
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
        return mse, r2

    def save_model(self):
        """
        Save the trained model and scaler.
        """
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print(f"Model and scaler saved to {self.model_path} and 'models/scaler.pkl'")

    def run_pipeline(self, data_filepath):
        """
        Run the entire training pipeline.
        """
        # Load and preprocess data
        data = self.load_data(data_filepath)
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Train the model
        print("Training the model...")
        self.train_model(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating the model...")
        self.evaluate_model(X_test, y_test)
        
        # Save the model
        self.save_model()

if __name__ == "__main__":
    # Replace 'data/dataset.csv' with your actual dataset path
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline('data/dataset.csv')
