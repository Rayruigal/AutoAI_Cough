import os


class DefaultConfig(object):
    # model config
    # mode = "train_dnn_keras"
    mode = "train_dnn_autokeras"
    # mode = "train_ml_autogluon"

    # save and load
    save_model_root = None

    # dataset related
    dataset_root = "./data/"
    # dataset name
    dataset_name = "acoustic"

    # training data & testing data split
    seed = 7  # None or number
    val_rate_new = 1.0/5


    # DNN train, test related
    worker_nums = 4   # how many workers for loading data
    batch_size = 32

    train_epochs = 10
    train_lr = 1e-4
