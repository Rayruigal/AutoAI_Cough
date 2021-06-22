# converts csv acoustic dataset to csv format suitable for AutoAI (OCL, IBM, Neunets)
# new format: [features],[label] -> x0,x1,...,x1023,y0

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

    # labels
    class_names = ['snort', 'snore_nose', 'cough_explosive', 'cough_soft', 'wheeze', 'snore_throat', 'clearing_throat']

    # data (csv)
    features = pd.read_csv("./arrayfeatures_WL617_WS200_FL46_FS9_NF16_NT64_log_mel_spect.csv")

    #
    feature_array = features["feature_array"].values

    X = []
    #for i in range(10):
    for i in range(len(features)):
        x = string_to_numpy(feature_array[i])
        X.append(x)

    X = np.array(X)
    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    print("X.shape=", X.shape)

    Y_gt, classes_unique = pd.factorize(features['class'])
    print("classes_unique=", classes_unique)

    df = pd.DataFrame(data=X)
    df.columns = ['x'+str(i) for i in range(X.shape[1])]
    df['y0'] = Y_gt

    print(df)

    df.to_csv('data_acoustic.csv', index=False, float_format='%.6f')
