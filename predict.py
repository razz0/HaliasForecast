"""Command line tool for trying out the prediction models"""

import argparse

from sklearn.externals import joblib
from haliasdata import get_bird_species

from models import prediction_models, load_models

parser = argparse.ArgumentParser(description='Predict traffic disruptions')
parser.add_argument('var1', help='variable 1', type=float)
parser.add_argument('var2', help='variable 2', type=float)
parser.add_argument('var3', help='variable 3', type=float)
parser.add_argument('var4', help='variable 4', type=float)
args = parser.parse_args()

input_scaler = joblib.load('model/scaler.pkl')

species = get_bird_species()

load_models(prediction_models)

for model in prediction_models:
    #print 'Model %s' % (model.model)
    value_tuple = input_scaler.transform((args.var1, args.var2, args.var3, args.var4))
    prediction = model.predict(value_tuple)
    print('Model %s: %s' % (model.name, prediction))
