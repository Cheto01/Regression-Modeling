import argparse
from models.train_model import train_model, MODELS
from absl import logging
logging.set_verbosity(logging.INFO)
parser = argparse.ArgumentParser(description='Train a machine learning model and Perform predictions using a trained model.')
parser.add_argument('--train', default=False, type=bool, help='we want to train the model')

parser.add_argument('--model', type=str, default='RF', nargs='+', help=f'Specify the model name from {MODELS}, can be multiple models', required=True)
parser.add_argument('--crossval', type=bool, default=True, help='Perform cross-validation')
parser.add_argument('--data', type=str, default='data/raw/woven_data.tsv', help='Path to the dataset')
parser.add_argument('--model_path', type=str, help='Path to the trained model file')

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


if __name__ == '__main__':
    print(args)
    if not args.model_path and not args.train:
        logging.info('no model path passed for the inference the trained Logistic Regression Model will be used')
        args.model_path ='models/LR.pkl'
    if args.sex:
        print('noo')
    print(args)