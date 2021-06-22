import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

class IdentityScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array

    def inverse_transform(self, input_array, y=None):
        return input_array


def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def remove_columns(dataframe, input_list):
    df = dataframe[input_list]
    to_remove = []
    for d in df.columns:
        u = len(df[d].unique())
        if u <= -1:
            to_remove.append(d)

    active_input_list = list(set(input_list)^set(to_remove))
    return active_input_list
    #c2 = np.array(c)
    #idx = np.where(c2 <= 1)[0]
    # print(idx)
    # print(len(idx))



class DataSet:

    def __init__(self, datafile, input_list, output_list, input_scaler=None, output_scaler=None, dataset_type="regression", delim=" ", feature_eng=False, is_train=True, data_shape=None):

        self.data_shape = data_shape

        self.input_list = input_list
        self.output_list = output_list
        self.dataset_type = dataset_type

        # load dataset
        if delim == " ":
            delim_whitespace = True
        else:
            delim_whitespace = False

        self.dataframe = pd.read_csv(datafile, delim_whitespace=delim_whitespace, header=None, skiprows=1, comment='#')
        #self.dataframe = pd.read_csv(datafile, delim_whitespace=delim_whitespace, header=None, skiprows=0, comment='#')
        #self.dataframe = pd.read_csv(datafile, delim_whitespace=delim_whitespace, skiprows=0, comment='#')

        ### Feature engineering
        if feature_eng:
            input_list = remove_columns(self.dataframe, self.input_list)
            self.input_list = input_list
        ###

        def get_sklearn_scaler(scaler):
            if scaler is None or scaler == 'Standard':
                return StandardScaler()
            elif scaler == 'MinMax':
                return MinMaxScaler()
            elif scaler == 'MaxAbs':
                return MaxAbsScaler()
            elif scaler == 'Robust':
                return RobustScaler()
            elif scaler == 'Identity':
                return IdentityScaler()

        self.input_scaler = get_sklearn_scaler(input_scaler)
        self.output_scaler = get_sklearn_scaler(output_scaler)

        tmp_inp = self.input_scaler.fit(self.dataframe[input_list])
        tmp_out = self.output_scaler.fit(self.dataframe[output_list])

    def process(self, test_size=0.20):
        # percentage of validation set (0.20: 20% validation size, 80% train size)
        # test_size = 0.20

        if test_size == 0:
            train = self.dataframe
            test = train
        else:
            (train, test) = train_test_split(self.dataframe, test_size=test_size, random_state=43)

        trainX = self.input_scaler.transform(train[self.input_list].values)
        testX = self.input_scaler.transform(test[self.input_list].values)

        trainY = self.output_scaler.transform(train[self.output_list].values)
        testY = self.output_scaler.transform(test[self.output_list].values)

        scaler_filename = "input_scaler.save"
        joblib.dump(self.input_scaler, scaler_filename)

        scaler_filename = "output_scaler.save"
        joblib.dump(self.output_scaler, scaler_filename)

        print("trainX.shape=", trainX.shape)
        self.input_dim = trainX.shape[1]

        print("trainY.shape=", trainY.shape)
        self.output_dim = trainY.shape[1]

        if self.output_dim == 1 and self.dataset_type == "classification":
            le = LabelEncoder()
            le.fit(trainY)
            print("le.classes_=", le.classes_)
            self.output_dim = len(le.classes_)
            self.classes = le.classes_

            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse=False, categories='auto')
            classes = self.classes.tolist()
            print("classes=", classes)
            classes = to_matrix(classes, 1)

            ohe.fit(classes)
            self.ohe = ohe
        else:
            self.ohe = None

        # acoustic
        if self.data_shape:
            total_shape = (-1,) + self.data_shape
            trainX = trainX.reshape(total_shape)
            testX = testX.reshape(total_shape)

        self.trainX = np.asarray(trainX)
        self.trainY = np.asarray(trainY)
        self.testX = np.asarray(testX)
        self.testY = np.asarray(testY)


    def inverse_transform_output(self, y):
        y_real = self.output_scaler.inverse_transform(y)

        return y_real

    def transform_output_proba(self, y):
        if self.ohe is not None:
            y_proba = self.ohe.transform(y)
        else:
            y_proba = None

        return y_proba

