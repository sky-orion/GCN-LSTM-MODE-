import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class S2sTransformer(nn.Module):

    def __init__(self,vocab_size,position_enc,d_model = 512,nhead = 8,num_encoder_layers=6,
                 num_decoder_layers=6,dim_feedforward=2048,dropout=0.1):
        super(S2sTransformer,self).__init__()

        # Preprocess
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder_src = position_enc(d_model=512)
        # tgt
        self.pos_encoder_tgt = position_enc(d_model=512)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_encoder_layers,encoder_norm)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm)
        self.output_layer = nn.Linear(d_model,vocab_size)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead


    def forward(self, src,tgt,src_mask = None,tgt_mask = None,
                memory_mask = None,src_key_padding_mask = None,
                tgt_key_padding_mask = None,memory_key_padding_mask = None):

        # word embedding
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # shape check
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # position encoding
        src = self.pos_encoder_src(src)
        tgt = self.pos_encoder_tgt(tgt)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_layer(output)
        # return output
        return softmax(output,dim = 2)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)
transformer_model = S2sTransformer(nhead=16, num_encoder_layers=12)