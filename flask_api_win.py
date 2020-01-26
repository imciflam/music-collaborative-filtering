from flask import Flask, json, request
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import pickle
from numba import jit
from random import random
import threading
import time
import sys

app = Flask(__name__)


@app.route('/knn', methods=['GET', 'POST'])
def get_closest_groups():
    closest_groups = print_artist_recommendations(
        request.json, wide_artist_data_zero_one, model_nn_binary, k=5)
    return json.dumps(closest_groups)


def background_calculation():
    # here goes some long calculation
    global wide_artist_data_zero_one
    wide_artist_data_zero_one = data_processing()
    global model_nn_binary
    model_nn_binary = pickle.load(
        open('finalized_model_short.sav', 'rb'))
    global result
    result = 42


@jit(parallel=True)
def data_processing():
    rus_data = pd.read_csv('short_well.csv')
    wide_artist_data_pivoted = rus_data.pivot(
        index='artist-name', columns='users', values='plays')
    wide_artist_data = wide_artist_data_pivoted.fillna(0)
    # applying the sign function in numpy to each column in the dataframe
    wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
    return wide_artist_data_zero_one

 
def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    query_index = None
    list_of_closest_groups = []
    ratio_tuples = []
    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    try:
        # get the index of the best artist match in the data
        query_index = max(ratio_tuples, key=lambda x: x[1])[2]
    except:
        print('No match. Try again')
        return None

    distances, indices = knn_model.kneighbors(
        artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=k + 1)

    for i in range(0, len(distances.flatten())):
        if i != 0:
            list_of_closest_groups.append(
                (artist_plays_matrix.index[indices.flatten()[i]]))
    return list_of_closest_groups


@app.route('/')
def get_empty():
    return("base route")


result = None


def main():
    thread = threading.Thread(target=background_calculation)
    print('background_calculation start')
    thread.start()
    # wait here for the result to be available before continuing
    thread.join()
    print('The result is', result)
    if result == 42:
        print('background_calculation is completed')
    else:
        print('background_calculation failed')
        sys.exit(0)


if __name__ == '__main__':
    main()
    app.run()
