
# evaluate the som model based on the training set

from minisom_3 import MiniSom
import numpy as np
from solution import solution
from AdjustedMinisom import AdjustedMinisom
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,accuracy_score


def optimized_som(train_inputs, train_outputs, test_inputs, test_outputs, Finalsolution, map_dim, epochs):

    print('now will do the final training...')

    num_of_features = np.shape(train_inputs)[1]  # number of features in the inputs

    som = AdjustedMinisom(map_dim, map_dim, num_of_features, sigma=1.0, learning_rate=0.5)

    som.new_final_weights_init(Finalsolution,train_inputs,map_dim)

    som.train_batch(train_inputs.values, epochs)

    print('done with the final training')

    ######
    ######
    ######
    train_outputs.unique()


    # Now Plotting the SOMS to see the results
    data_reduced = []
    for i, (t, vec) in enumerate(zip(train_outputs.values, train_inputs.values)):
        winnin_position = som.winner(vec)
        data_reduced.append(winnin_position)




    df = pd.DataFrame(data_reduced)

    uo = list(test_outputs.unique())
    len(uo)

    # Predicting Test based on the cell in which it is predicted
    wtest = []
    for vec in test_inputs.values:
        winnin_position = som.winner(vec)
        mask = np.logical_and(df[0] == winnin_position[0], df[1] == winnin_position[1])
        if mask.sum() == 0:
            l = np.random.randint(0, 9)
            wtest.append(uo[l])
        else:
            wtest.append(train_outputs[mask].mode()[0])

    predicted = wtest
    true = test_outputs.values

    f1 = f1_score(true, predicted, average='weighted')
    print("F1_Score: ", f1)
    p1 = precision_score(true, predicted, average='weighted')
    print("Precision: ", p1)
    r1 = recall_score(true, predicted, average='weighted')
    print("Recall :", r1)
    accu = accuracy_score(true, predicted)
    print("Accuracy : %.3f" % accu)