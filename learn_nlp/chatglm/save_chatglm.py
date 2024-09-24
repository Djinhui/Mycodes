import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
model = AutoModel.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True).half().cuda() # .cpu()

response, history = model.chat(tokenizer, '你好', history=[])
print(response)

response, history = model.chat(tokenizer, 'xxx', history=history)
print(response)

torch.save(model.state_dict(), './huggingface_saver/chatglm6b.pth')