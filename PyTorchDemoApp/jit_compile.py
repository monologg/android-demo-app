import torch
import numpy as np
from transformers import ElectraTokenizer, ElectraConfig
from model import ElectraForSequenceClassification


tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-finetuned-sentiment")

# Tokenize input text
text = "이 영화 왜 아직도 안 봄?"
output = tokenizer.encode_plus(
    text,
    return_tensors="pt",
)
print(output)

model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-finetuned-sentiment", torchscript=True)
model.eval()

traced_model = torch.jit.trace(
    model,
    output["input_ids"]
)
torch.jit.save(traced_model, "traced_model.pt")

# Load model
loaded_model = torch.jit.load("traced_model.pt")
loaded_model.eval()
with torch.no_grad():
    outputs = loaded_model(output["input_ids"])
print(outputs)
with torch.no_grad():
    outputs = model(output["input_ids"])
print(outputs)
