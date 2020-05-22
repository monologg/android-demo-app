import tensorflow as tf
from transformers import ElectraTokenizer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model import TFElectraForSequenceClassification

MAX_SEQ_LEN = 20

tokenizer = ElectraTokenizer.from_pretrained("monologg/electra-small-finetuned-imdb")
model = TFElectraForSequenceClassification.from_pretrained("monologg/electra-small-finetuned-imdb",
                                                           from_pt=True)

input_spec = tf.TensorSpec([1, MAX_SEQ_LEN], tf.int64)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

# Tokenize input text
text = "This movie is awesome lol!"
encode_inputs = tokenizer.encode_plus(
    text,
    return_tensors="tf",
    max_length=MAX_SEQ_LEN,
    pad_to_max_length=True
)

outputs = model(encode_inputs["input_ids"])
print(outputs[0])

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For normal conversion:
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]

# For conversion with FP16 quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_types = [tf.float16]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter = True

# For conversion with hybrid quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.experimental_new_converter = True

tflite_model = converter.convert()

open("app/src/main/assets/imdb_small.tflite", "wb").write(tflite_model)
