import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
from transformers import ElectraTokenizer

from model import TFElectraForSequenceClassification

parser = argparse.ArgumentParser()
# NOTE This should be same as the setting of the android!!!
parser.add_argument("--max_seq_len", default=40, type=int, help="Maximum sequence length")
args = parser.parse_args()

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-sentiment")
model = TFElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-finetuned-sentiment",
                                                           from_pt=True)

input_spec = tf.TensorSpec([1, args.max_seq_len], tf.int64)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

# Tokenize input text
text = "이 영화 왜 아직도 안 봄?"
encode_inputs = tokenizer.encode_plus(
    text,
    return_tensors="tf",
    max_length=args.max_seq_len,
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

open("app/src/main/assets/nsmc_small.tflite", "wb").write(tflite_model)
