from flask import Flask, json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from fuzzywuzzy import fuzz
import pickle

knn = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

api = Flask(__name__)


@api.route('/knn', methods=['GET'])
def get_companies():
    rus_data = pd.read_table('well.tsv')
    # artists = rows, users = columns
    wide_artist_data = rus_data.pivot(
        index='artist-name', columns='users', values='plays').fillna(0)
    # applying the sign function in numpy to each column in the dataframe
    wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
    model_nn_binary = pickle.load(
        open('nn_model.sav', 'rb'))
    closest_groups = print_artist_recommendations(
        'The Prodigy', wide_artist_data_zero_one, model_nn_binary, k=10)
    return json.dumps("")


def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    query_index = None
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

    return None


if __name__ == '__main__':
    api.run()
