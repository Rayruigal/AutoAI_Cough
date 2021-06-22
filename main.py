import os
import sys
import time
import numpy as np

from plot_csv import analyze_results, analyze_results_class
from dataset import DataSet
from model import KerasModel, AutogluonModel
from config import DefaultConfig


# MAIN CODE ###
def main():
    if len(sys.argv) == 3:
        datafile = sys.argv[1]
        test_datafile = sys.argv[2]
    elif len(sys.argv) == 2:
        datafile = sys.argv[1]
        test_datafile = None
    else:
        datafile = None
        test_datafile = None

    opt = DefaultConfig()

    # random state initialization
    if opt.seed is not 'None':
        random_seed = opt.seed
    else:
        random_seed = int(time.time())
    np.random.seed(random_seed)

    # dataset file
    if datafile is None:
        datafile = opt.dataset_root + opt.dataset_name + '/train_data_acoustic.csv'
        test_datafile = opt.dataset_root + opt.dataset_name + '/test_data_acoustic.csv'

    print("datafile=", datafile)
    print("test_datafile=", test_datafile)

    use_conv1d = False
    dataset_kwargs = {}
    model_kwargs = {}
    # load dataset
    if 'napoli' in datafile:
        input_list = list(range(0, 3))
        output_list = list(range(3, 4))
        scaler = 'Standard'
        output_scaler = 'Standard'
        dataset_type="regression"
        use_conv1d = False
    elif 'acoustic' in datafile:
        input_list = list(range(0, 1024))
        output_list = list(range(1024, 1025))
        scaler = "Standard"
        output_scaler = "Identity"
        dataset_type = "classification"
        if opt.mode == 'train_dnn_keras':
            # init_model = 'acoustic_model.json'
            init_model = 'my_acoustic_model2.json'
            #init_model = None
            if init_model is None:
                data_shape = None
            else:
                data_shape = (16,64,1)

            model_kwargs = {'init_model': init_model}
        elif opt.mode == 'train_dnn_autokeras':
            data_shape = (16,64,1)
            model_kwargs = {'use_conv': True}
        else:
           data_shape = None
        dataset_kwargs = {'data_shape': data_shape}
    else:
        raise Exception("Unknown dataset")

    data = DataSet(datafile, input_list, output_list, input_scaler=scaler, output_scaler=output_scaler, dataset_type=dataset_type, delim=",", feature_eng=False, **dataset_kwargs)
    input_list = data.input_list
    test_data = DataSet(test_datafile, input_list, output_list, input_scaler="Identity", output_scaler="Identity", dataset_type=dataset_type, delim=",", is_train=False, **dataset_kwargs)
    test_data.input_scaler = data.input_scaler
    test_data.output_scaler = data.output_scaler

    use_test_as_val = False
    if use_test_as_val:
        opt.val_rate_new = 0.0
    else:
        opt.val_rate_new = 0.2

    data.process(test_size=opt.val_rate_new)
    test_data.process(test_size=0)

    if use_test_as_val:
        data.testX = test_data.testX
        data.testY = test_data.testY

    if opt.mode == 'train_dnn_keras' or opt.mode == 'train_dnn_autokeras':
        model = KerasModel(data.input_dim, data.output_dim, dataset_type=dataset_type, method=opt.mode, conv1d=use_conv1d, **model_kwargs)
    elif opt.mode == 'train_ml_autogluon':
        model = AutogluonModel(data.input_dim, data.output_dim, dataset_type=dataset_type, method=opt.mode)
    else:
        raise Exception("Unknown mode")

    # train the model
    model.fit(data)

    # make predictions on the testing data
    preds = model.predict(test_data.testX)

    # reconstruct output values and compute statistics
    if dataset_type == "regression":
        testY = data.inverse_transform_output(test_data.testY)
        predY = data.inverse_transform_output(preds)
    elif dataset_type == "classification":
        predY_proba = model.predict_proba(test_data.testX)
        testY_proba = model.transform_output_proba(test_data.testY)
        testY = model.transform_output(test_data.testY)
        testY = testY.flatten()
        predY = preds

    # analyze results
    if dataset_type == "regression":
        R2values = []
        for i in range(data.output_dim):
            R2, mae = analyze_results(testY, predY, i, str(i), verbose=False, savefile=True)
            print('test set:', i, R2, mae)
            R2values.append([i, R2])
        np.savetxt("r2_{}.csv".format(os.getpid()), R2values, fmt='%d %.6f', delimiter=" ")

    elif dataset_type == "classification":
        from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
        print(classification_report(testY, predY))
        print("roc_auc_score=", roc_auc_score(testY_proba, predY_proba))
        print("confusion matrix=\n", confusion_matrix(testY, predY))

        analyze_results_class(testY, predY, 0, str(0), verbose=False, savefile=True)

    # save the model
    model.save(opt.save_model_root)

    best_params = model.best_params()
    print("best_params=", best_params)


if __name__ == "__main__":
    main()
