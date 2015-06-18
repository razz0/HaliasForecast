'''Generate prediction models'''

import math
import iso8601
import argparse

import numpy as np

from rdflib import Graph, RDF, Namespace
from rdflib.collection import Collection
import models

ns_qb = Namespace('http://purl.org/linked-data/cube#')
ns_hs = Namespace('http://ldf.fi/schema/halias/')
ns_bio = Namespace('http://www.yso.fi/onto/bio/')


parser = argparse.ArgumentParser(description='Generate models')
#parser.add_argument('generate', help='Generate normal models', type=bool)
parser.add_argument('-o', help='Generate optimized models (slow)',
                    dest='optimized', action='store_const', const=True, default=False)
parser.add_argument('-v', help='Add verbosity',
                    dest='verbose', action='store_const', const=True, default=False)
args = parser.parse_args()



def preprocess_data(input_datacube, output_datacube, input_dimensions, output_dimension, link_dimension):
    '''
    Format data cubes to numpy arrays.
    :param input_datacube: Graph containing only data cube input observations
    :type input_datacube: Graph
    :param output_datacube:
    :type output_datacube: Graph
    :param input_dimensions:
    :param output_dimension:

    :rtype tuple of np.array
    '''
    xx = {}
    yy = {}

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

        output_value = None
        for output_observation in output_datacube.subjects(link_dimension, link_resource):
            if output_datacube.objects(output_observation, ns_hs.observedSpecies) != ns_bio.FMNH_372767:
                # TODO: Remove this
                continue
            output_value = next(output_datacube.objects(output_observation, output_dimension))

        # TODO: Try imputing missing values: http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py

        # Skipping rows with missing values for now
        if link_resource is not None \
                and output_value is not None \
                and not any([math.isnan(val) for val in good_values]):
            xx[link_resource] = good_values
            yy[link_resource] = output_value

    x = np.array([values for index, values in sorted(xx.items())], dtype='float_')
    y = np.array([values for index, values in sorted(yy.items())], dtype='int_')

    return x, y



# TODO: Remove "hs:haliasObservationDay  false" observations from input cube when preprocessing

input_cube = Graph()
input_cube.parse('../../HALIAS/JavaOutput/halias_weather_cube.ttl', format='turtle')

# TODO: Create separate model for each species (separate model selection?)

output_cube = Graph()
output_cube.parse('../../HALIAS/JavaOutput/HALIAS4_full.ttl', format='turtle')

x_train, y_train = preprocess_data(input_cube,
                                   output_cube,
                                   [ns_hs.rainfall, ns_hs.standardTemperature],
                                   ns_hs.countTotal,
                                   ns_hs.refTime)

def _save_model(generated_model):
    generated_model.save_model()
    print('Saved model %s - %s' % (generated_model.name, generated_model.model))

for model in models.prediction_models:

    # if model.name == 'Optimized Forest':
    #     if args.optimized:
    #         best_score = 0
    #         best_params = {}
    #         for n_estimators in range(2, 50):
    #             if args.verbose:
    #                 print "Calculating models for %s trees" % n_estimators
    #             for criterion in ['gini', 'entropy']:
    #                 for max_features in range(1, 7):  # + ['auto', 'log2']:
    #                     for max_depth in range(3, 15) + [None]:
    #                         for class_weight in [None]:  # ['auto', None]:
    #                             model.model_kwargs = dict(n_estimators=n_estimators,
    #                                                       criterion=criterion,
    #                                                       max_features=max_features,
    #                                                       max_depth=max_depth,
    #                                                       class_weight=class_weight)
    #                             model.generate_model(x_train, y_train)
    #                             score = model.model.score(x_test, y_test)
    #                             if score > best_score:
    #                                 best_params = model.model_kwargs
    #                                 best_score = score
    #                                 print "%s -- %s" % (score, best_params)
    #
    #         model.model_kwargs = best_params
    #         print "Best found params: %s" % best_params
    #         # Best found params:
    #         # {'max_features': 2, 'n_estimators': 38, 'criterion': 'gini', 'max_depth': 10, 'class_weight': None}
    #         print "Feature importances: %s" % model.model.feature_importances_
    #         # Feature importances: [ 0.08076559  0.30474923  0.14358273  0.19469095  0.1400464   0.1361651 ]
    #         model.generate_model(x_train, y_train)
    #         _save_model(model)
    # else:
    #     if not args.optimized:
    model.generate_model(x_train, y_train)
    _save_model(model)

