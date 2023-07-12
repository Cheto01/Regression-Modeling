import numpy as np
import pickle
import argparse
from data.data_processing import 

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_user_input(model, user_input):
    # Preprocess user input to match the format expected by the model
    # Here you can perform any necessary data transformations or feature engineering

    # Convert user input to a numpy array
    input_array = np.array([user_input])

    # Perform prediction
    predictions = model.predict(input_array)

    return predictions

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform predictions using a trained model.')
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

    # Print the predictions
    print("Predictions:", predictions)
