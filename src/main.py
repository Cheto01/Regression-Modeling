import argparse
import numpy as np
from train_model import train_model
from features.build_features import get_training_testing_data, scale_feature
import pandas as pd
from absl import flags, logging

FLAGS = flags.FLAGS

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a machine learning model.')
parser.add_argument('--train', type=bool, help='we want to train the model')
parser.add_argument('--model', type=str, default='RF', help='Specify the model name')
parser.add_argument('--crossval', action='store_true', help='Perform cross-validation')
parser.add_argument('--data', type=str, help='Path to the dataset')parser = argparse.ArgumentParser(description='Perform predictions using a trained model.')
parser.add_argument('--model_path', type=str, help='Path to the trained model file', required=True)
parser.add_argument('--sex', type=int, help='Sex attribute')
parser.add_argument('--lang', type=str, help='Language attribute')
parser.add_argument('--country', type=str, help='country attribute')
parser.add_argument('--age', type=int, help='Age attribute')
parser.add_argument('--first', type=str, help='First name')
parser.add_argument('--last', type=str, help='Last name attribute')
parser.add_argument('--hours_studied', type=float, help='studied hours attribute')
parser.add_argument('--dojo_class', type=bool, help='Dojo class taken attribute')
parser.add_argument('--test_prep', type=bool, help='Test preparation')
parser.add_argument('--test_pass', type=bool, help='Pass or not')
parser.add_argument('--notes', type=str, help='Notes attribute')



args = parser.parse_args()

# Load the trained model
model = load_model(args.model_path)

# Get user input
user_input = [args.sex, args.lang, args.country, args.age, args.first, args.last, args.hours_studied, args.dojo_class, args.test_prep, args.test_pass, args.notes]


# Perform prediction
predictions = predict_user_input(model, user_input)


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
