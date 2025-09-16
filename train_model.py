
"""
Crop Recommendation System - Training Script
============================================

This script trains machine learning models for crop recommendation based on 
soil nutrients and climate conditions.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationTrainer:
    def __init__(self):
        self.encoder = None
        self.scaler = None
        self.model = None
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    def load_data(self, filepath='Crop_recommendation.csv'):
        """Load and validate the dataset"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Unique crops: {len(self.df['label'].unique())}")
            print(f"Crop types: {list(self.df['label'].unique())}")

            # Check for missing values
            missing_values = self.df.isnull().sum().sum()
            if missing_values > 0:
                print(f"⚠️  Warning: {missing_values} missing values found")
                # Handle missing values
                self.df = self.df.dropna()
                print(f"Rows after removing missing values: {len(self.df)}")

            return True
        except FileNotFoundError:
            print(f"❌ Error: Dataset file '{filepath}' not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def preprocess_data(self):
        """Preprocess the data - encoding and scaling"""
        try:
            print("\n🔄 Preprocessing data...")

            # Encode labels
            self.encoder = LabelEncoder()
            self.df['label_encoded'] = self.encoder.fit_transform(self.df['label'])
            print(f"✓ Label encoding completed. Classes: {self.encoder.classes_}")

            # Prepare features and target
            X = self.df[self.feature_columns]
            y = self.df['label_encoded']

            # Check for any invalid values
            if X.isnull().any().any():
                print("⚠️  Warning: NaN values found in features")
                X = X.fillna(X.mean())  # Fill with mean values

            # Scale features
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            print("✓ Feature scaling completed")

            return X_scaled, y
        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            return None, None

    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        try:
            print("\n🤖 Training models...")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Testing set: {X_test.shape[0]} samples")

            # Define models
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
            }

            best_model = None
            best_accuracy = 0
            best_model_name = ""

            # Train and evaluate each model
            for name, model in models.items():
                print(f"\n--- Training {name} ---")

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy:.4f}")

                # Store best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name

                # Print detailed results for best models
                if accuracy > 0.95:  # Only show detailed results for high-performing models
                    print("Classification Report:")
                    print(classification_report(y_test, y_pred, target_names=self.encoder.classes_, zero_division=0))

            print(f"\n🏆 Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
            self.model = best_model
            return True

        except Exception as e:
            print(f"❌ Error in model training: {str(e)}")
            return False

    def save_model(self, model_dir='models'):
        """Save the trained model components"""
        try:
            print(f"\n💾 Saving model components to '{model_dir}'...")

            # Create model directory if it doesn't exist
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Save components
            with open(f'{model_dir}/encoder.pkl', 'wb') as f:
                pickle.dump(self.encoder, f)

            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            with open(f'{model_dir}/model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            print("✅ Model saved successfully!")
            print("Files saved: encoder.pkl, scaler.pkl, model.pkl")
            return True

        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False

    def test_prediction(self):
        """Test the prediction function with sample data"""
        try:
            print("\n🧪 Testing prediction function...")

            # Test cases with expected results
            test_cases = [
                {"input": (90, 42, 43, 20.88, 82.00, 6.50, 202.94), "expected": "rice"},
                {"input": (80, 45, 20, 25.0, 70.0, 6.5, 80.0), "expected": "maize"},
                {"input": (25, 70, 80, 18.0, 16.0, 7.0, 75.0), "expected": "chickpea"},
                {"input": (5, 80, 200, 22.0, 92.0, 6.5, 110.0), "expected": "apple"}
            ]

            for i, test_case in enumerate(test_cases):
                N, P, K, temp, hum, ph, rain = test_case["input"]

                # Create input dataframe
                input_data = pd.DataFrame({
                    'N': [N], 'P': [P], 'K': [K],
                    'temperature': [temp], 'humidity': [hum],
                    'ph': [ph], 'rainfall': [rain]
                })

                # Scale input
                input_scaled = self.scaler.transform(input_data)

                # Predict
                prediction = self.model.predict(input_scaled)
                crop_name = self.encoder.inverse_transform(prediction)[0]

                print(f"Test {i+1}: N={N}, P={P}, K={K}, T={temp}°C, H={hum}%, pH={ph}, R={rain}mm")
                print(f"Predicted: {crop_name} | Expected: {test_case['expected']}")
                print(f"✓ {'Correct' if crop_name == test_case['expected'] else 'Different prediction'}\n")

            return True

        except Exception as e:
            print(f"❌ Error in testing: {str(e)}")
            return False

def main():
    """Main training pipeline"""
    print("="*50)
    print("🌾 CROP RECOMMENDATION SYSTEM - TRAINING")
    print("="*50)

    # Initialize trainer
    trainer = CropRecommendationTrainer()

    # Load data
    if not trainer.load_data():
        return

    # Preprocess data
    X, y = trainer.preprocess_data()
    if X is None:
        return

    # Train models
    if not trainer.train_models(X, y):
        return

    # Save model
    if not trainer.save_model():
        return

    # Test prediction
    trainer.test_prediction()

    print("\n" + "="*50)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\n🚀 Next steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Use the web interface to get crop recommendations")

if __name__ == "__main__":
    main()
