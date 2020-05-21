import torch
import numpy as np
from transformers import ElectraTokenizer, ElectraConfig
from model import ElectraForSequenceClassification

# 1. Convert model

MAX_SEQ_LEN = 20  # NOTE This should be same as the setting of the android!!!

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-sentiment")

model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-finetuned-sentiment", torchscript=True)
model.eval()

input_ids = torch.tensor([[0] * MAX_SEQ_LEN], dtype=torch.long)
print(input_ids.size())
traced_model = torch.jit.trace(
    model,
    input_ids
)
torch.jit.save(traced_model, "app/src/main/assets/nsmc_small.pt")

# 2. Testing...
# Tokenize input text
text = "이 영화 왜 아직도 안 봄?"
encode_inputs = tokenizer.encode_plus(
    text,
    return_tensors="pt",
    max_length=MAX_SEQ_LEN,
    pad_to_max_length=True
)
print(encode_inputs)
print(encode_inputs["input_ids"].size())

# Load model
loaded_model = torch.jit.load("app/src/main/assets/nsmc_small.pt")
loaded_model.eval()
with torch.no_grad():
    outputs = loaded_model(encode_inputs["input_ids"])
print(outputs)
with torch.no_grad():
    outputs = model(encode_inputs["input_ids"])
print(outputs)
