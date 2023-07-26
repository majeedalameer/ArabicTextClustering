from minisom import MiniSom
import numpy as np
import minisom
from solution import solution

from solution import solution

class AdjustedMinisom(MiniSom):

    def __init__(self, x, y, input_len, sigma, learning_rate, decay_function=minisom.asymptotic_decay,
                 neighborhood_function='gaussian', random_seed=None):
        super().__init__(x, y, input_len, sigma, learning_rate, decay_function, neighborhood_function, random_seed)



    # here i replaced the num_of_clusters parameter with the map dimension
    def new_weights_init(self, test_weights,inputs,map_dim):
        """Initializes the weights from solution
        """
        print("adjusting the weights for MiniSom")
        print("shape of weights:")
        print(self._weights)
        print(self._weights.shape)
        print('above is the shape of original weights')

        num_of_features = np.shape(inputs)[1]  # number of features in the inputs

        ###
        ###
        ###
        #my code
        new_weights = test_weights.reshape(map_dim, map_dim, num_of_features)  # readjust the weight
        ###
        ###
        ###
        #new_weights = x.reshape(num_of_clusters, num_of_features)  # readjust the weight
        self._weights = new_weights
        print("Done from adjusting the weights")
        print(self._weights.shape)
        print('above is the shape of the updated weights')


    def new_final_weights_init(self, solution,inputs,map_dim):
        """Initializes the weights from solution
        """
        print("adjusting the weights for MiniSom")
        print("shape of weights:")
        print(self._weights)
        print(self._weights.shape)
        print('above is the shape of original weights')

        num_of_features = np.shape(inputs)[1]  # number of features in the inputs
        x =  solution.bestIndividual
        new_weights = x.reshape(map_dim, map_dim, num_of_features)  # readjust the weight
        ###
        ###
        ###
        #new_weights = x.reshape(num_of_clusters, num_of_features)  # readjust the weight
        self._weights = new_weights
        print("Done from adjusting the weights")
        print(self._weights.shape)
        print('above is the shape of the updated weights')
