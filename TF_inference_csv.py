# performs inference using the original csv file

import sys
import tensorflow as tf
import numpy as np

# the following libraries are NOT needed for inference, its only for preparing example testing data
import pandas as pd


def string_to_numpy(text, dtype=None):
    """
    Convert text into 1D or 2D arrays using np.matrix().
    The result is returned as an np.ndarray.
    """
    import re
    text = text.strip()
    # Using a regexp, decide whether the array is flat or not.
    # The following matches either: "[1 2 3]" or "1 2 3"
    is_flat = bool(re.match(r"^(\[[^\[].+[^\]]\]|[^\[].+[^\]])$",
                            text, flags=re.S))
    # Replace newline characters with semicolons.
    text = text.replace("]\n", "];")
    # Prepare the result.
    result = np.asarray(np.matrix(text, dtype=dtype))
    return result.flatten() if is_flat else result


if __name__ == "__main__":

    # user index
    if len(sys.argv) > 1:
        sample_index = int(sys.argv[1])
    else:
        sample_index = 0

    # labels
    class_names = ['snort', 'snore_nose', 'cough_explosive', 'cough_soft', 'wheeze', 'snore_throat', 'clearing_throat']

    # load TFLite model
    model_path = 'saved_model/ETH/TF/TestUserID_16.h5'
    model = tf.keras.models.load_model(model_path)
    model.summary()


    # load example test data
    # to be replaced with audio recording and processing (output is a log-mel spectrogram representation of shape 16x64x1)
    # features = pd.read_pickle("./arrayfeatures_WL617_WS200_FL46_FS9_NF16_NT64_log_mel_spect.pickle")

    features = pd.read_csv("./data/arrayfeatures_WL617_WS200_FL46_FS9_NF16_NT64_log_mel_spect.csv")
    feature_array = features["feature_array"].values
    X = []
    for i in range(len(features)):
    #for i in range(10):
        x = string_to_numpy(feature_array[i])
        X.append(x)

    X = np.array(X)
    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print("X.shape=", X.shape)

    Y_gt, classes_unique = pd.factorize(features['class'])
    print("classes_unique=", classes_unique)

    print("inference for sample index:", sample_index) 
    test_index = sample_index   # anything within the range

    X_test = X[test_index]
    Y_test = Y_gt[test_index]
    #print("X_test=", X_test)
    print("X_test.shape=", X_test.shape)
    print("Y_test=", Y_test)

    # inference
    predict_probabilities = model.predict(X_test.reshape(1,X_test.shape[0],X_test.shape[1],X_test.shape[2]))
    predict_label = class_names[np.argmax(predict_probabilities)]

    print('The tested audio clip is classified as of the sound: ' + predict_label)

    print('The tested audio clip is classified as each of the pre-defined sound categories with the following probabilities: ')
    for i in range(len(class_names)):
        print(class_names[i] + ': ' + str(predict_probabilities[0][i]))
