'''
Helper functions for working with linked Halias dataset
'''

from rdflib import Graph, Namespace, RDF, URIRef
from sklearn.externals import joblib


ns_qb = Namespace('http://purl.org/linked-data/cube#')
ns_hs = Namespace('http://ldf.fi/schema/halias/')
ns_bio = Namespace('http://www.yso.fi/onto/bio/')
ns_tr = Namespace('http://www.yso.fi/onto/taxonomic-ranks/')


def load_weather_cube():
    try:
        weather_cube = joblib.load('data/halias_weather_cube.pkl')
    except IOError:
        print('Input data pickle not found, reading from RDF graphs.')
        weather_cube = Graph()
        weather_cube.parse('data/halias_weather_cube.ttl', format='turtle')
        joblib.dump(weather_cube, 'data/halias_weather_cube.pkl')

    return weather_cube


def load_bird_cube():
    try:
        bird_cube = joblib.load('data/halias_bird_cube.pkl')
    except IOError:
        print('Output data pickle not found, reading from RDF graphs.')
        bird_cube = Graph()
        bird_cube.parse('data/HALIAS0_full.ttl', format='turtle')
        bird_cube.parse('data/HALIAS1_full.ttl', format='turtle')
        bird_cube.parse('data/HALIAS2_full.ttl', format='turtle')
        bird_cube.parse('data/HALIAS3_full.ttl', format='turtle')
        bird_cube.parse('data/HALIAS4_full.ttl', format='turtle')
        joblib.dump(bird_cube, 'data/halias_bird_cube.pkl')

    return bird_cube


def load_bird_ontology():
    bird_ontology = Graph()
    bird_ontology.parse('data/halias_taxon_ontology.ttl', format='turtle')

    return bird_ontology


def get_bird_species(prune_by_rarity=False):
    '''
    Get bird species as RDF resources

    About 170 common, 323 rare taxa
    :param prune_by_rarity:
    :return:
    '''

    bird_ontology = load_bird_ontology()

    species = bird_ontology.subjects(RDF.type, ns_tr.Species)

    if prune_by_rarity:
        species_list = [sp for sp in species if next(bird_ontology.objects(sp, ns_hs.rarity)) == URIRef(ns_hs[prune_by_rarity])]
    else:
        species_list = list(species)

    return species_list
