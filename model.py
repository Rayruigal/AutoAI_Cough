import os
import sys
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from config import DefaultConfig


opt = DefaultConfig()

if opt.mode == 'train_dnn_keras' or opt.mode == 'train_dnn_autokeras':
    print("importing tensorflow...")
    from keras_config import use_tfkeras
    from utils import compute_weights_memory, compute_params, compute_inference_memory, compute_weights_memory2, compute_inference_flops, get_model_memory_usage

    if use_tfkeras:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Flatten, Conv1D, Conv2D, Conv3D, MaxPooling3D, BatchNormalization
        from tensorflow.keras import optimizers, losses
        from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
        import tensorflow
    else:
        from keras.models import Sequential
        from keras.models import load_model, model_from_json
        from keras.layers import Dense, Dropout, MaxPooling1D, Flatten, Conv1D, Conv3D, MaxPooling3D
        from keras import optimizers, losses
        from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
        import keras


if opt.mode == 'train_dnn_autokeras':
    try:
        import autokeras as ak
        autokeras_available = True
    except:
        autokeras_available = False


if opt.mode == 'train_ml_autogluon':
    try:
        from autogluon import TabularPrediction as Task
        autogluon_available = True
    except:
        print("could not import autogluon")
        autogluon_available = False
        sys.exit(1)


def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


class BaseModel:
    __metaclass__ = ABCMeta

    def __init__(self, input_dim, output_dim, dataset_type="regression"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.dataset_type = dataset_type

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError


    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    def predict(self, x):
        # make predictions on the testing data
        print("BaseModel: predicting values ...")
        y_pred = self.model.predict(x)

        #print("y_pred=", y_pred)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        #if self.dataset_type == "classification":
        #   y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def predict_proba(self, x):
        # make predictions on the testing data
        print("predicting probs ...")
        y_pred = self.model.predict_proba(x)

        print("y_proba=", y_pred)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        #if self.dataset_type == "classification":
        #   y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def transform_output(self, y_pred):

        try:
            y_pred = self.model_ohe.transform(y_pred)
        except:
            pass

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if self.dataset_type == "classification":
            y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def transform_output_proba(self, y_pred):

        try:
            y_pred = self.model_ohe.transform(y_pred)
        except:
            pass

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def best_params(self):

        try:
            #print("cv_results=", self.model.cv_results_)
            #print("statistics =", self.model.sprint_statistics())
            #print("models=", self.model.show_models())
            #print("best params=", self.model.cv_results_['params'][self.model.best_index_])
            pass
        except:
            pass

        try:
            return self.model.best_params_
        except:
            return None


class KerasModel(BaseModel):

    def __init__(self, input_dim, output_dim, dataset_type, method='train_dnn_keras', init_model=None, conv1d=False, use_conv=False):
        super().__init__(input_dim, output_dim, dataset_type)
        self.method = method
        self.init_model = init_model
        self.conv1d = conv1d
        self.use_conv = use_conv
        if self.method == 'train_dnn_keras':
            self.__init_fx__(input_dim, output_dim, dataset_type)
        elif self.method == 'train_dnn_autokeras':
            self.__init_ak__(input_dim, output_dim, dataset_type)


    def fit(self, dataset):
        if self.method == 'train_dnn_keras':
            self.fit_fx(dataset)
        elif self.method == 'train_dnn_autokeras':
            self.fit_ak(dataset)


    def __init_ak__(self, input_dim, output_dim, dataset_type):

        metrics=["mean_absolute_error"]  # "mean_squared_error"]

        tmp_path = "/tmp/age"
        os.system('rm -rf /tmp/age')

        if self.dataset_type == "regression":
            # model = ak.StructuredDataRegressor(overwrite=True, max_trials=100, directory=tmp_path, objective='val_loss', metrics=['mae'], tuner="bayesian")
            input_node = ak.Input()
            output_node = input_node
            # output_node = ak.Normalization()(output_node)
            output_node = ak.DenseBlock(num_layers=3, dropout=0.0, use_batchnorm=False)(output_node)
            output_node = ak.RegressionHead(dropout=0.0, metrics=['mae'])(output_node)

            model = ak.AutoModel(inputs=input_node, outputs=output_node, directory=tmp_path, max_trials=10,
                                 objective="val_loss", tuner="bayesian")  # "greedy" "random"
        else: # "classification"

            if not self.use_conv:
                # model = ak.StructuredDataClassifier(overwrite=True, max_trials=10, num_classes=output_dim, objective='val_accuracy', tuner="bayesian")
                input_node = ak.Input()
                output_node = input_node
                # output_node = ak.Normalization()(output_node)
                output_node = ak.DenseBlock(num_layers=3, dropout=0.0, use_batchnorm=False)(output_node)
                output_node = ak.ClassificationHead(multi_label=True, dropout=0.0, metrics=['accuracy'])(output_node)
            else:
                # model = ak.ImageClassifier(overwrite=True, max_trials=10, objective='val_accuracy')  #max_model_size=70000000)
                input_node = ak.ImageInput()
                output_node = input_node
                output_node = ak.ConvBlock()(output_node)
                output_node = ak.ClassificationHead(multi_label=True)(output_node)
                # output_node = ak.ClassificationHead(multi_label=True, dropout=0.0, metrics=['accuracy'])(output_node)

            model = ak.AutoModel(inputs=input_node, outputs=output_node, directory=tmp_path, max_trials=50,
                                 objective="val_accuracy", tuner="hyperband")  # "greedy" "random"

        self.model = model


    def __init_fx__(self, input_dim, output_dim, dataset_type):

        if self.init_model is not None:
            """
            init_model = self.init_model
            #init_model.summary()
            #config = init_model.get_config()
            #model = keras.Model.from_config(config)
            model = self.init_model
            """
            # load json and create model
            json_file = open(self.init_model, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
        else:
            # create model
            model = Sequential()

            # choose weight initializer (not important)
            #initializer = 'he_uniform'
            initializer = 'normal'

            # choose neural network architecture: number of layers, neurons per layer, activation function
            if self.dataset_type == "regression":
                if not self.conv1d:
                    if output_dim > 10:
                        batch_normalization = False
                        dropout = False

                        model.add(Dense(256, input_dim=input_dim, kernel_initializer=initializer, activation='relu'))
                        if batch_normalization: model.add(BatchNormalization())
                        if dropout: model.add(Dropout(0.25))

                        model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                        if batch_normalization: model.add(BatchNormalization())
                        if dropout: model.add(Dropout(0.25))

                        model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                        if batch_normalization: model.add(BatchNormalization())
                        if dropout: model.add(Dropout(0.25))

                        model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                        if batch_normalization: model.add(BatchNormalization())
                        if dropout: model.add(Dropout(0.25))

                        model.add(Dense(output_dim, kernel_initializer=initializer, activation="linear"))
                    else:
                        model.add(Dense(256, input_dim=input_dim, kernel_initializer=initializer, activation='relu'))
                        model.add(Dense(128, kernel_initializer=initializer, activation='relu'))
                        model.add(Dense(64, kernel_initializer=initializer, activation='relu'))
                        model.add(Dense(32, kernel_initializer=initializer, activation='relu'))
                        model.add(Dense(output_dim, kernel_initializer=initializer, activation="linear"))
                else:
                    model.add(Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(input_dim, 1)))
                    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
                    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
                    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
                    # model.add(Dropout(0.1))
                    # model.add(MaxPooling1D(pool_size=2))
                    model.add(Flatten())
                    #model.add(Dense(64,kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(32, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(16, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="linear"))
            else:
                if not self.conv1d:
                    model.add(Dense(256, input_dim=input_dim, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="softmax"))
                else:
                    model.add(Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=(input_dim, 1)))
                    model.add(Conv1D(filters=8, kernel_size=1, activation='relu'))
                    model.add(Dropout(0.1))
                    model.add(MaxPooling1D(pool_size=2))
                    model.add(Flatten())
                    #model.add(Dense(128,kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(64, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(32, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="softmax"))

        # choose optimizer
        opt = optimizers.Adam()

        # choose loss function
        if self.dataset_type == "regression":
            loss = losses.mean_absolute_error
        else:
            loss='categorical_crossentropy'

        # Compile the model
        metrics=[]
        if self.dataset_type == "classification":
            metrics.append('accuracy')

        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        model.summary()

        self.model = model


    @staticmethod
    def _lr_schedule(epoch):
        # CHOICE 1: learning rate schedule
        # lr = 1.0e-3
        lr = opt.train_lr
        factor = 10
        if epoch > 40*factor:
            lr *= 0.5e-3
        elif epoch > 30*factor:
            lr *= 1e-3
        elif epoch > 20*factor:
            lr *= 1e-2
        elif epoch > 10*factor:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def fit_data_ak(self, trainX, trainY, testX, testY, input_list=None):
        # lr_scheduler = LearningRateScheduler(self._lr_schedule)
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
        callbacks = [es] # lr_scheduler]

        # train the model
        # choose number of epochs and batch_size
        epochs = opt.train_epochs
        batch_size = opt.batch_size

        print("training automodel...")

        self.model.fit(x=trainX, y=trainY, epochs=epochs, validation_data=(testX, testY), callbacks=callbacks)

        # self.model.fit(x=trainX, y=trainY, epochs=epochs)
        exported_model = self.model.export_model()
        print(exported_model)
        exported_model.summary()
        print("Evaluating...")
        self.model.evaluate(testX, y=testY)
        print("Evaluated...")
 
        if 1:
            del self.model
            self.model = exported_model 
        else:
            # serialize model to JSON
            os.system('rm -f ak_model.json ak.model.h5')
            model_json = exported_model.to_json()
            with open("ak_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            exported_model.save("ak.model.h5")

            del self.model
            self.model = tensorflow.keras.models.load_model('ak.model.h5', custom_objects=ak.CUSTOM_OBJECTS)

    def fit_ak(self, dataset):

        # train the model
        # choose number of epochs and batch_size

        print("training model...")
        trainX = dataset.trainX
        trainY = dataset.trainY
        testX = dataset.testX
        testY = dataset.testY

        if self.dataset_type == "classification":
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse=False, categories='auto')
            classes = dataset.classes.tolist()
            print("classes=", classes)
            classes = to_matrix(classes, 1)

            ohe.fit(classes)

            trainY = ohe.transform(trainY)
            print("train categories=", ohe.categories_)

            testY = ohe.transform(testY)
            print("test categories=", ohe.categories_)
            self.model_ohe = ohe
 
        self.fit_data_ak(trainX, trainY, testX, testY)


    def fit_data_fx(self, trainX, trainY, testX, testY, input_list=None):
        lr_scheduler = LearningRateScheduler(self._lr_schedule)

        if self.dataset_type == "classification":
            monitor='val_accuracy'
        else:
            monitor='val_loss'

        es = EarlyStopping(monitor='val_loss', verbose=1, patience=100)

        ckpt = ModelCheckpoint('keras_model_ckpt_{}.h5'.format(os.getpid()), monitor=monitor, verbose=1, save_best_only=True, mode='auto')
        callbacks = [lr_scheduler, ckpt]

        # train the model
        # choose number of epochs and batch_size
        epochs = opt.train_epochs
        batch_size = opt.batch_size

        print("training model...")

        if True:
            trainY = trainY.astype(float)
            testY = testY.astype(float)
            history = self.model.fit(trainX, trainY, validation_data=(testX, testY),
                                    epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)


        try:
            ohe = self.model_ohe
            del self.model
            self.model = load_model('keras_model_ckpt_{}.h5'.format(os.getpid()))
            self.model_ohe = ohe
        except BaseException as e:
            print("exception: ", str(e))
            del self.model
            self.model = load_model('keras_model_ckpt_{}.h5'.format(os.getpid()))


    def fit_fx(self, dataset):
        print("training model...")
        trainX = dataset.trainX
        trainY = dataset.trainY
        testX = dataset.testX
        testY = dataset.testY

        if self.dataset_type == "classification":
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse=False, categories='auto')
            classes = dataset.classes.tolist()
            print("classes=", classes)
            classes = to_matrix(classes, 1)

            ohe.fit(classes)

            trainY = ohe.transform(trainY)
            print("train categories=", ohe.categories_)

            testY = ohe.transform(testY)
            print("test categories=", ohe.categories_)
            self.model_ohe = ohe


        if self.conv1d:
            trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
            testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

        self.fit_data_fx(trainX, trainY, testX, testY, input_list=dataset.input_list)

    def fit_data(self, trainX, trainY, testX, testY, input_list=None):
        if self.method == 'train_dnn_keras':
            return self.fit_data_fx(trainX, trainY, testX, testY, input_list)
        elif self.method == 'train_dnn_autokeras':
            return self.fit_data_ak(trainX, trainY, testX, testY, input_list)

    def predict(self, x):
        # make predictions on the testing data
        print("predicting values ...")

        if self.conv1d:
            x = x.reshape((x.shape[0], x.shape[1], 1))

        if self.dataset_type == "classification":
            # y_pred = self.model.predict_classes(x)
            y = self.model.predict(x)
            y_pred = np.argmax(y, axis=-1)
        else:
            y_pred = self.model.predict(x)


        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred


    def predict_proba(self, x):
        # make predictions on the testing data
        # print("predicting values (proba) ...")

        if self.conv1d:
            x = x.reshape((x.shape[0], x.shape[1], 1))

        y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred


    def transform_output(self, y_pred):

        try:
            y_pred = self.model_ohe.transform(y_pred)
        except:
            pass

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if self.dataset_type == "classification":
           y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def transform_output_proba(self, y_pred):

        try:
            y_pred = self.model_ohe.transform(y_pred)
        except:
            pass

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def save(self, path):
        if self.method == 'train_dnn_keras':
            self.model.save('keras_model_{}.h5'.format(os.getpid()))
        else:
            self.model.save('keras_model_{}.tf'.format(os.getpid()), save_format='tf')


class AutogluonModel(BaseModel):

    def __init__(self, input_dim, output_dim, dataset_type, method='train_ml_autogluon', config=None):

        self.method = method
        self.savedir = None
        self.config = config if config else {}
        super().__init__(input_dim, output_dim, dataset_type)
        if dataset_type == "regression":
            if output_dim > 1:
                raise NotImplementedError

        self.model = None

    def fit(self, dataset):

        print("training Autogluon model...")

        trainX = dataset.trainX
        trainY = dataset.trainY
        testX = dataset.testX
        testY = dataset.testY

        if self.dataset_type == "classification":
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse=False, categories='auto')
            classes = dataset.classes.tolist()
            print("classes=", classes)
            classes = to_matrix(classes, 1)

            ohe.fit(classes)
            self.model_ohe = ohe

            trainY = ohe.transform(trainY)
            trainY = np.argmax(trainY, axis=-1)
            print("train categories=", ohe.categories_)
            testY = ohe.transform(testY)
            testY = np.argmax(testY, axis=-1)
            print("test categories=", ohe.categories_)
            self.model_ohe = ohe
        else:
            self.ohe = None

        df_x = pd.DataFrame(data=trainX)
        df_y = pd.DataFrame(data=trainY)
        df = pd.concat([df_x, df_y], axis=1, ignore_index=True)
        label_column = len(df.columns)-1

        train_data = Task.Dataset(data=df)
        savedir = 'ag_models_{}/'.format(os.getpid()) # where to save trained models
        self.savedir = savedir

        # self.model = Task.fit(train_data=train_data, label=label_column, output_directory=savedir, eval_metric='r2')
        # results = self.model.fit_summary()
        auto_stack = self.config.get("auto_stack", False)
        time_limits = self.config.get("time_limits", 120)
        if self.dataset_type == "classification":
            self.model = Task.fit(train_data=train_data, label=label_column, output_directory=savedir,
                                  problem_type='multiclass',
                                  excluded_model_types=['NN', 'CAT', 'GBM'],
                                  auto_stack=auto_stack, time_limits=time_limits,
                                  keep_only_best=True)

        else:
            # https://auto.gluon.ai/api/autogluon.task.html, autogluon.tabular.TabularPrediction.fit
            # available_metrics = ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error',
            # 'median_absolute_error', 'r2']
            self.model = Task.fit(train_data=train_data, label=label_column, output_directory=savedir,
                                  problem_type='regression', eval_metric='mean_absolute_error',
                                  excluded_model_types=['NN', 'CAT', 'GBM'],
                                  auto_stack=auto_stack, time_limits=time_limits,
                                  keep_only_best=True)
            # nthreads_per_trial=1
            # not used: hyperparameter_tune=False, num_trials=100, search_strategy = search_strategy
        _ = self.model.fit_summary(verbosity=1)

    def predict(self, x):
        # make predictions on the testing data

        print("autogluon: predicting values ...")
        df_x = pd.DataFrame(data=x)
        test_data = Task.Dataset(data=df_x)

        y_pred = self.model.predict(test_data)
        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def predict_proba(self, x):
        # make predictions on the testing data

        print("autogluon: predicting proba...")
        df_x = pd.DataFrame(data=x)
        test_data = Task.Dataset(data=df_x)

        y_pred = self.model.predict_proba(test_data)
        # print("y_pred=", y_pred)
        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    """
    def transform_output_proba(self, y_pred):
        try:
            y_pred = self.model_ohe.transform(y_pred)
        except:
            pass

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def transform_output(self, y_pred):

        try:
            y_pred = self.model_ohe.transform(y_pred)
        except:
            pass

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if self.dataset_type == "classification":
            y_pred = np.argmax(y_pred, axis=-1)

        return y_pred
    """


    def save(self, path):
        if path:
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(self.savedir, path)
            # os.rename(self.savedir, path)

