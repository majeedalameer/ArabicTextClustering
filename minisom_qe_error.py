import pandas as pd
from minisom import MiniSom
import numpy as np
from AdjustedMinisom import AdjustedMinisom
from solution import solution

def minisom_qe_error(solution, inputs, map_dim):

    number_of_features = np.shape(inputs)[1] #number of features in the inputs

    print('network created, now will adjust the weights')

    som = AdjustedMinisom(map_dim, map_dim, number_of_features, sigma=1.0, learning_rate=0.5)
    #add my weights
    som.new_weights_init(solution,inputs,map_dim)

    print('weights added, will train the model')
    som.train_batch(inputs.values, 10)

    err = som.quantization_error(inputs.values)

    print(err)

    return err
