import argparse
import numpy as np
from train_model import train_model
from features.build_features import get_training_testing_data, scale_feature
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a machine learning model.')
parser.add_argument('--model', type=str, default='RF', help='Specify the model name')
parser.add_argument('--crossval', action='store_true', help='Perform cross-validation')
parser.add_argument('--data', type=str, help='Path to the dataset')
args = parser.parse_args()

# Load and preprocess the data
df = pd.read_csv(args.data)
X_train, X_test, y_train, y_test = get_training_testing_data(df, target='pass')

# Scale the features if needed
X_train = scale_feature(X_train)
X_test = scale_feature(X_test)

# Train the model
model_name = args.model
model = train_model(model_name, X_train, y_train, cross_validation=args.crossval)

# Print the trained model and cross-validation scores
print(f"Trained Model: {model}")
if isinstance(model, tuple):
    trained_model, scores = model
    print("Cross-validation scores:")
    for mn, score in scores.items():
        print(f"{mn}: {np.mean(score):.4f} (±{np.std(score):.4f})")
else:
    print("Cross-validation score:", model)
