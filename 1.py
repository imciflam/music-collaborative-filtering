from random import random
import threading
import time
from flask import Flask, json, request
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from fuzzywuzzy import fuzz
import pickle
from numba import jit

result = None


def background_calculation():
    print('background_calculation')
    # here goes some long calculation
    time.sleep(5)

    # when the calculation is done, the result is stored in a global variable
    global result
    result = 42
    print('result')


def main():
    thread = threading.Thread(target=background_calculation)
    print('thread start')
    thread.start()
    # wait here for the result to be available before continuing
    thread.join()
    print('The result is', result)


if __name__ == '__main__':
    main()
