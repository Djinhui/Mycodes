import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)    #64*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    #64*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))    #256   model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)   #64*1*512

    def forward(self, x):     #[seq,batch,d_model]
        return x + self.pe[:x.size(0), :]   #64*64*512

class TransAm(nn.Module):
    def __init__(self, feature_size,out_size,d_model=512, num_layers=1, dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding=nn.Linear(feature_size,512)
        self.pos_encoder = PositionalEncoding(d_model)          #50*512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, out_size)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding):
        #shape of src  [seq,batch,feature_size]
        if self.src_key_padding_mask is None:
            mask_key = src_padding   #[batch,seq]
            self.src_key_padding_mask = mask_key
        src=self.embedding(src)        #[seq,batch,d_model]
        src = self.pos_encoder(src)    #[seq,batch,d_model]
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output = self.decoder(output)  #[seq,batch,output_size]
        self.src_key_padding_mask = None
        return output



