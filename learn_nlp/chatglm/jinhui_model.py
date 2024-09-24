import torch
import torch.nn as nn
from huggingface_saver import configuration_chatglm
from huggingface_saver import modeling_chatglm
from transformers import AutoTokenizer
from huggingface_saver import tokenization_chatglm


class JinhuiModel(nn.Module):
    def __init__(self, model_path='./huggingface_saver', config=None, strict=True):
        super.__init__()
        self.glm_model = modeling_chatglm.ChatGLMForConditionalGeneration(config=config)
        model_dict = torch.load(model_path)
        self.glm_model.load_state_dict(model_dict,strict=strict)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def fowward(self, input_ids, labels=None, position_ids=None, attention_mask=None):
        logits, hidden_states = self.glm_model.forward(input_ids=input_ids, position_ids=None, attention_mask=None)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:,1:].contiguous()

            logits_1 = shift_logits.view(-1, shift_logits.size(-1))
            logits_2 = shift_labels.view(-1)

            loss = self.loss_fct(logits_1, logits_2)
        return logits, hidden_states, loss
    

    def generate(self, start_question_text='抗原呈递的原理是什么?', continue_seq_length=128, tokenizer=None, temperature=0.95, top_p=0.95):
        if '：' not in start_question_text:
            input_text_ori = start_question_text
            input_text = f'[Round 0]\n问：{input_text_ori}\n答'
        else:
            input_text = start_question_text

        input_ids = tokenizer.encode(input_text)

        for _ in range(continue_seq_length):
            input_ids_tensor = torch.tensor([input_ids]).to('cuda')
            logits, _, _ = self.forward(input_ids_tensor)
            logits = logits[:, -3]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)
            input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
            if next_token.item() == 130005:
                break

        result = tokenizer.decode(input_ids)
        return result
    

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, 'ds_numel'):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"trainable params:{trainable_params} || all params:{all_param} || trainable%:{100*trainable_params/all_param}")


if __name__ == '__main__':
    config = configuration_chatglm.ChatGLMConfig()
    model = JinhuiModel(config=config).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True, cache_dir="./huggingface_saver")
    inputs_text_ori = "抗原呈递的原理是什么？"
    result = model.generate(inputs_text_ori, continue_seq_length=256, tokenizer=tokenizer)
    print(result)

    while True:
        print('请输入：')
        que = input()
        inputs_text_ori = que
        result = model.generate(inputs_text_ori, continue_seq_length=256, tokenizer=tokenizer)
        print(result)

        



