import argparse
import numpy as np
from src.models.train_model import train_model, MODELS
from src.models.predict import predict, predict_user_input, load_model
from src.features.build_features import get_training_testing_data, scale_feature, get_prediction_data
import pandas as pd
from src.data.data_processing import import_data, feature_selection
from absl import flags, logging
import os

FLAGS = flags.FLAGS
logging.set_verbosity(logging.INFO)

def _verify_flags(args):
    if args.sexe and args.lang and args.country and args.age and \
            args.hours_studied and args.dojo_class and args.test_prep:
        raise ValueError("all the features needs to be \
            provided in order to make the classification of one person \
            during the call of the main function, you can otherwise use \
            a csv file where you put the inference data")
    #if not (args.model_path and args.train):
    #    logging.info('no model path passed for the inferece the trained Logistic Regression Model will be used')
    #    args.model_path ='models/LR.pkl'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a machine learning model \
    and Perform predictions using a trained model.')
parser.add_argument('--train', type=bool, help='we want to train the model')
parser.add_argument('--predict', type=bool, default=True, help='whether or not to make predictions')
parser.add_argument('--model', type=str, default='RF', nargs='+', help=f'Specify \
    the model name from {MODELS}, can be multiple models', required=True)
parser.add_argument('--crossval', type=bool, default=True, help='Perform cross-validation')
parser.add_argument('--scale_features', type=bool, default=False, help='Performs \
    scaling on age and hours_studied features')
parser.add_argument('--data', type=str, default='data/raw/woven_data.tsv', help='Path to the dataset')
# parser.add_argument('--model_path', type=str, default='models/LR.pkl', help='Path to the trained model file')
parser.add_argument('--output_path', type=str, default='reports/model_output/', help='Path to the trained model file')

#parser.add_argument('--sexe', type=str, help='Sex attribute, male/female')
#parser.add_argument('--lang', type=str, help='Language attribute')
#parser.add_argument('--country', type=str, help='country attribute')
#parser.add_argument('--age', type=int, help='Age attribute')
#parser.add_argument('--first', type=str, help='First name')
#parser.add_argument('--last', type=str, help='Last name attribute')
#parser.add_argument('--hours_studied', type=float, help='studied hours attribute')
#parser.add_argument('--dojo_class', type=bool, help='Dojo class taken attribute')
#parser.add_argument('--test_prep', type=bool, help='Test preparation')
#parser.add_argument('--pass', type=bool, help='Pass or not')
#parser.add_argument('--notes', type=str, help='Notes attribute')

args = parser.parse_args()

def main(args=args):
    print(args)
    if args.train:
        logging.info('In Training mode')
        df = import_data(args.data)
        X_train, X_test, y_train, y_test = get_training_testing_data(df, target='pass', scale_features = args.scale_features)

        if len(args.model) ==1:
            if args.crossval:
                model, scores = train_model(args.model[0], X_train, y_train, cross_validation=args.crossval)
            else:
                model = train_model(args.model[0], X_train, y_train, cross_validation=args.crossval)
        else:
            if args.crossval:
                model, scores = train_model(args.model, X_train, y_train, cross_validation=args.crossval)
            else:
                model = train_model(args.model, X_train, y_train, cross_validation=args.crossval)
        if args.predict:
            predictions, (precision, recall, f1, support), accuracy = predict(model, X_test, y_test)
            print(accuracy)
            os.makedirs(args.output_path+args.model[0], exist_ok=True)
            pd.DataFrame(np.array([precision, recall, f1, support]).T,
                            columns=['precision', 'recall', 'f_score', 'support']).to_csv(args.output_path+'_'.join(args.model)+'_metrics.csv')
            pd.DataFrame(np.array([predictions,y_test]).T,
                            columns=['predictions', 'true_value']).to_csv(args.output_path+'_'.join(args.model)+'_predictions.csv')

    elif args.predict:
        model = load_model(args.model)
        df = import_data(args.data)
        X_test, y_test = get_prediction_data(df, target='pass', scale_features=args.scale_features)
        predictions, (precision, recall, f1, support), accuracy = predict(model, X_test, y_test)
        pd.DataFrame(np.array([precision, recall, f1, support, accuracy]),
                         columns=['precision', 'recall', 'f_score', 'support', 'accuracy']).to_csv(args.output_path+'_'.join(args.model)+'_metrics.csv')
        pd.DataFrame(np.array([predictions,y_test]).T,
                         columns=['predictions', 'true_value']).to_csv(args.output_path+'_'.join(args.model)+'_predictions.csv')

        
if __name__ == '__main__':
    main(args)
            
# Get user input
# user_input = [args.sex, args.lang, args.country, args.age, args.first, args.last, args.hours_studied, args.dojo_class, args.test_prep, args.test_pass, args.notes]


# Perform prediction
#predictions = predict_user_input(model, user_input)
#
#
## Load and preprocess the data
#df = pd.read_csv(args.data)
#X_train, X_test, y_train, y_test = get_training_testing_data(df, target='pass')
#
## Scale the features if needed
#X_train = scale_feature(X_train)
#X_test = scale_feature(X_test)
#
## Train the model
#model_name = args.model
#model = train_model(model_name, X_train, y_train, cross_validation=args.crossval)
#
## Print the trained model and cross-validation scores
#print(f"Trained Model: {model}")
#if isinstance(model, tuple):
#    trained_model, scores = model
#    print("Cross-validation scores:")
#    for mn, score in scores.items():
#        print(f"{mn}: {np.mean(score):.4f} (±{np.std(score):.4f})")
#else:
#    print("Cross-validation score:", model)
#