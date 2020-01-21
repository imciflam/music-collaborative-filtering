from sanic import Sanic, response
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from fuzzywuzzy import fuzz
import pickle
from numba import jit

app = Sanic(__name__)


@app.route("/start", methods=["GET"])
async def get_groups(request):
    # return json({ "hello": "world" })
    return response.text('200')


@app.route("/users", methods=["POST"])
def create_user(request):
    return response.text(request.body)


app.run(host="0.0.0.0", port=8000)
