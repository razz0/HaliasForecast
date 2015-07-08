'''Generate prediction models'''
from collections import defaultdict
import math
import argparse

import numpy as np
from rdflib import Graph, RDF, URIRef
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import cross_validation

from haliasdata import *
import models

parser = argparse.ArgumentParser(description='Generate models')
#parser.add_argument('generate', help='Generate normal models', type=bool)
parser.add_argument('-o', help='Generate optimized models (slow)',
                    dest='optimized', action='store_const', const=True, default=False)
parser.add_argument('-v', help='Add verbosity',
                    dest='verbose', action='store_const', const=True, default=False)
args = parser.parse_args()


# TODO: Aggregate daily sums to some classes

def preprocess_data(input_datacube, output_datacube, input_dimensions, output_dimension, link_dimension, separate_models=None):
    '''
    Format data cubes to numpy arrays.

    :param input_datacube: Graph containing only data cube input observations
    :type input_datacube: Graph
    :param output_datacube:
    :type output_datacube: Graph
    :param input_dimensions:
    :param output_dimension:

    '''
    xx = {}
    yy = defaultdict(dict)

    # TODO: Allow mapping of input dimensions to 1 or more values, eg. refTime -> day of year

    for input_observation in input_datacube.subjects(RDF.type, ns_qb.Observation):
        good_values = [None] * len(input_dimensions)
        link_resource = None

        for dimension, value in input_datacube.predicate_objects(input_observation):
            if dimension in input_dimensions:
                # print('%s - %s' % (dimension, value))
                good_values[input_dimensions.index(dimension)] = value.toPython()

                # fmi_values = map(float, (fmi_data[timestamp].get('r_1h'), fmi_data[timestamp].get('t2m'), fmi_data[timestamp].get('ws_10min')))
                # if [x for x in fmi_values if math.isnan(x)]:
                #     continue
                # if fmi_values[0] == -1.0:
                #     fmi_values[0] = 0  # Assuming "-1.0" rainfall means zero rain
                # if not str(hsl_data[timestamp]).isdigit():
                #     print "HSL %s" % hsl_data[timestamp]
                #     continue
                #
                # if use_hour:
                #     obs_time = iso8601.parse_date(timestamp)
                #     fmi_values += [obs_time.hour]
                # if extra_date_params:
                #     fmi_values += [obs_time.isoweekday()]
                #     fmi_values += [obs_time.month]
                #
                # xx.append(fmi_values)
                # yy.append(int(float(hsl_data.get(timestamp))))

            if dimension == link_dimension:
                link_resource = value

        output_values = {}
        model_name = None
        for output_observation in output_datacube.subjects(link_dimension, link_resource):
            if separate_models:
                model_name = str(next(output_datacube.objects(output_observation, ns_hs.observedSpecies)))

            output_values[model_name] = next(output_datacube.objects(output_observation, output_dimension)).toPython()

        # TODO: Try imputing missing values: http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py

        # Skipping rows with missing values for now
        if link_resource is not None \
                and not any([math.isnan(val) for val in good_values]):
            xx[link_resource] = good_values
            # yy[link_resource] = output_value
            for model_name, output in output_values.items():
                yy[model_name].update({link_resource: output})

    x = np.array([values for index, values in sorted(xx.items())], dtype='float_')

    y_dict = {}
    for model_name, model_y in yy.items():
        y = np.array([values for index, values in sorted(model_y.items())], dtype='int_')
        y_dict[model_name] = y

    input_scaler = StandardScaler()

    x = input_scaler.fit_transform(X=x)
    # y = StandardScaler().fit_transform(X=y)

    return x, y_dict, input_scaler


# TODO: Remove "hs:haliasObservationDay  false" observations from input cube when preprocessing

# TODO: Try 0-inflated models

weather_cube = load_weather_cube()
bird_cube = load_bird_cube()

bird_ontology = load_bird_ontology()

species = bird_ontology.subjects(RDF.type, ns_tr.Species)
species = [sp for sp in species if next(bird_ontology.objects(sp, ns_hs.rarity)) == URIRef(ns_hs.common)]
species = species[:10]  # TODO Use all

# TODO: Create separate model for each species (separate model selection?)


x_train, y_train_dict, scaler = preprocess_data(weather_cube,
                                bird_cube,
                                [ns_hs.rainfall, ns_hs.standardTemperature, ns_hs.airPressure, ns_hs.cloudCover],
                                ns_hs.countTotal,
                                ns_hs.refTime,
                                separate_models={URIRef(ns_hs.observedSpecies): species})

joblib.dump(scaler, 'model/scaler.pkl')

for model_name, y_train in y_train_dict.items():

    best_model = None
    best_score = 0

    for model in models.prediction_models:

        model.generate_model()
        scores = cross_validation.cross_val_score(model.model, x_train, y_train, cv=4)

        if scores.mean() > best_score:
            best_model = model

    if best_model.model:

        # print("Model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("%s model %s accuracy: %0.2f (+/- %0.2f)" % (model_name, model.name, scores.mean(), scores.std() * 2))

    best_model.fit_model(x_train, y_train)
    best_model.filename = model_name

    best_model.save_model()
    print('Saved model %s - %s' % (best_model.name, best_model.model))

