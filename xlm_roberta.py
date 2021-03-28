import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

input_ids = torch.tensor([tokenizer("Hello world ")['input_ids']])

print(model(input_ids, output_hidden_states=True, output_attentions=True))