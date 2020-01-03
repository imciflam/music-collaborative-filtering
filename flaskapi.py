from flask import Flask, json, request
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from fuzzywuzzy import fuzz
import pickle
import time
from numba import jit


app = Flask(__name__)


@app.route('/knn', methods=['GET', 'POST'])
def get_groups():
    # print(request.json)  # getting json
    start = time.process_time()
    rus_data = pd.read_csv('well.tsv', sep='\t')
    print(time.process_time() - start)
    # artists = rows, users = columns
    wide_artist_data_zero_one = data_processing(rus_data)
    print(time.process_time() - start)
    model_nn_binary = pickle.load(
        open('nn_model.sav', 'rb'))
    closest_groups = print_artist_recommendations(
        request.json[0], wide_artist_data_zero_one, model_nn_binary, k=10)
    return json.dumps(closest_groups)


@jit()
def data_processing(rus_data):
    wide_artist_data = rus_data.pivot(
        index='artist-name', columns='users', values='plays').fillna(0)
    # applying the sign function in numpy to each column in the dataframe
    wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
    return wide_artist_data_zero_one


def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    query_index = None
    # list1 = {}
    list_of_closest_groups = []
    ratio_tuples = []
    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))

    print('Possible matches: {0}\n'.format(
        [(x[0], x[1]) for x in ratio_tuples]))

    try:
        # get the index of the best artist match in the data
        query_index = max(ratio_tuples, key=lambda x: x[1])[2]
    except:
        print('No match. Try again')
        return None

    distances, indices = knn_model.kneighbors(
        artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(
                artist_plays_matrix.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(
                i, artist_plays_matrix.index[indices.flatten()[i]], distances.flatten()[i]))
            # list1[artist_plays_matrix.index[indices.flatten()[i]]] = distances.flatten()[
            #    i]
            list_of_closest_groups.append(
                (artist_plays_matrix.index[indices.flatten()[i]]))
    return list_of_closest_groups

    # return None


@app.route('/')
def get_empty():
    return("")


if __name__ == '__main__':
    app.run()
