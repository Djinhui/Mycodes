import torch
from config.config import Config
from model.CoupletModel import CoupletModel
from utils.data_helper import my_tokenizer, LoadCoupletDataset



def greedy_decode(model, src, max_len, start_symbol, config, data_laoder):
    src = src.to(config.device)
    memory = model.encoder(src)

    ys = torch.ones(1,1).fill_(start_symbol).type(torch.long).to(config.device)
    for i in range(max_len-1):
        memory = memory.to(config.device)
        out = model.decoder(ys,memory) # [tgt_len,1,embed_dim]
        out = out.transpose(0,1) # [1,tgt_len, embed_dim]
        prob = model.classification(out[:, -1])  # 只对对预测的下一个词进行分类
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == data_laoder.EOS_IDX:
            break

    return ys


def couplet(model, src, data_loader, config):
    vocab = data_loader.vocab
    tokenizer = data_loader.tokenizer
    model.eval()
    tokens = [vocab.stoi[tok] for tok in tokenizer(src)]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    tgt_tokens = greedy_decode(model, src, max_len=num_tokens+5, start_symbol=data_loader.BOS_IDX, config=config,
                               data_loader=data_loader).flatten()
    return ''.join([vocab.itos[tok] for tok in tgt_tokens]).replace('<bos>', '').replace('<eos>','')



def do_couplet(srcs, config):
    data_loader = LoadCoupletDataset(config.train_corpus_file_paths,
                                     batch_size=config.batch_size,
                                     tokenizer=my_tokenizer,
                                     min_freq=config.min_freq)
    couplet_model = CoupletModel(vocab_size=len(data_loader.vocab),
                                 d_model=config.d_model,
                                 nhead=config.num_head,
                                 num_encoder_layers=config.num_encoder_layers,
                                 num_decoder_layers=config.num_decoder_layers,
                                 dim_feedforward=config.dim_feedforward,
                                 dropout=config.dropout)
    couplet_model = couplet_model.to(config.device)
    loaded_paras = torch.load(config.model_save_dir + '/model.pkl')
    couplet_model.load_state_dict(loaded_paras)
    results = []
    for src in srcs:
        r = couplet(couplet_model, src, data_loader, config)
        results.append(r)

    return results


if __name__ == '__main__':
    srcs = ["晚风摇树树还挺",
            "忽忽几晨昏，离别间之，疾病间之，不及终年同静好",
            "风声、雨声、读书声，声声入耳",
            "上海自来水来自海上"]
    tgts = ["晨露润花花更红",
            "茕茕小儿女，孱羸若此，娇憨若此，更烦二老费精神",
            "家事、国事、天下事，事事关心",
            ""]
    config = Config()
    srcs = [" ".join(src) for src in srcs]

    results = do_couplet(srcs, config)
    for src, tgt, r in zip(srcs, tgts, results):
        print(f"上联：{''.join(src.split())}")
        print(f" AI: {r}")
        print(f"下联：{tgt}")
        print("=======")
