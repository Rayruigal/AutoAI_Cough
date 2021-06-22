import tensorflow as tf
import argparse
import os

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", nargs="?", required=True, help="Model file")
args = parser.parse_args()

print(args.file)

fname, fext = os.path.splitext(args.file)


if fext == '.json':
    json_file = open(args.file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
else:
    loaded_model = tf.keras.models.load_model(args.file)


loaded_model.summary()
