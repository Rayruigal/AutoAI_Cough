# Modes and methods

Available in config.py

- "train_dnn_keras"  (keras)
- "train_dnn_autokeras" (autokeras)
- "train_ml_autogluon" (autogluon)

# Execution

python main.py ./data/acoustic/train_data_acoustic.csv ./data/acoustic/test_data_acoustic.csv


# Manual keras models

In main.py (line 52, 'acoustic')
            
- init_model = 'acoustic_model.json'     : custom CNN network (to be trained)  
- init_model = 'my_acoustic_model2.json' : custom CNN network (to be trained)
- init_model = None                      : fixed MLP network (defined in model.py) 


