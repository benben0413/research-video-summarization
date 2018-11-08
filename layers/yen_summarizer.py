import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from .lstmcell import StackedLSTMCell
from ipdb import set_trace

import torch.nn.functional as F

from tqdm_logger import seclog, log


def get_sinusoid_encoding_table(n_position=1519, d_hid=1024, padding_idx=0):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    # set_trace()
    return torch.FloatTensor(sinusoid_table)

class encoder_decoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 1024)
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid())

    def forward(self, features):


class attentive_encoder_decoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """Simple LSTM to summarize video"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, 1024)
        self.attention_dec = Decoder()
        self.out = nn.Sequential(
            nn.Linear(1024, 1 ),
            nn.Sigmoid())

    def forward(self, features, init_hidden=None, return_attns=False):
        attn_out = []
        out = torch.zeros(1, features.shape[1], 1).cuda()
        in_input = torch.zeros(1, 1, features.shape[2]).cuda()
        # in_h = torch.randn(1, features.shape[2]).cuda()
        # in_c = torch.randn(1, features.shape[2]).cuda()
        tmp, (hx, cx) = self.lstm(in_input)
        for i in range(features.shape[1]):
            att_feature, *attn = self.attention_dec(features, hx, return_attns)
            tmp2, (hx1, cx1) = self.lstm(att_feature, (hx, cx))
            hx = hx1
            cx = cx1
            scores = self.out(cx)
            out[:, i, :] = scores
            attn_out += [attn]

        if return_attns:
            return out, attn_out
        return out,


class simple_encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Simple LSTM to summarize video"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        # [1, seq, input_size]   
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2+input_size, 1 ),
            nn.Sigmoid())
        # out = [1, seq, 1]

    def forward(self, features, init_hidden=None):

        hidden_state, (_, _) = self.lstm(features.view(features.shape[1], 1, -1))
        cat_feature = torch.cat((features.view(features.shape[1], 1, -1), hidden_state), -1)
        scores = self.out(cat_feature.squeeze(0))
        
        return scores

class attentive_encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """Simple LSTM to summarize video"""
        super().__init__()
        self.lstm_fir = nn.LSTM(input_size, 512, num_layers, bidirectional=True)
        self.lstm = nn.LSTM(input_size, 512, num_layers, bidirectional=False)
        # [1, seq, input_size]   
        self.attention_enc = Encoder()
        self.out = nn.Sequential(
            nn.Linear(1024, 1 ),
            nn.Sigmoid())
        # self.out = nn.Sequential(
        #     nn.Sigmoid())
        # out = [1, seq, 1]

    def forward(self, features, init_hidden=None, return_attns=False):

        # hidden_state, (_, _) = self.lstm(features)
        # add_feature = torch.cat((hidden_state, features), -1)
        # features, (_, _) = self.lstm_fir(features.view(features.shape[1], 1, -1))
        cat_feature, *attn = self.attention_enc(features, return_attns)
        # cat_feature = self.attention_enc(pos, features)
        # out_feature, (_, _) = self.lstm(cat_feature.view(cat_feature.shape[1], 1, -1))
        scores = self.out(cat_feature)
        set_trace

        print(attn)
        input()

        if return_attns:
            return scores, attn

        return scores,

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    # def __init__(
    #         self,
    #         n_src_vocab, len_max_seq, d_word_vec,
    #         n_layers=1, n_head=8, d_k=64, d_v=64,
    #         d_model=512, d_inner=2048, dropout=0.1):
    def __init__(
            self,
            n_layers=3, n_head=8, d_k=64, d_v=64, #n_head=8
            d_model=1024, d_inner=512, dropout=0.1):

        super().__init__()

        # n_position = len_max_seq + 1

        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(),
        #     freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, return_attns=False):
    # def forward(self, src_seq, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        # slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        # non_pad_mask = get_non_pad_mask(src_seq)
        # set_trace()

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = src_seq
        # set_trace()
        # enc_output = src_seq + self.position_enc(src_pos)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers=2, n_head=4, d_k=64, d_v=64, #n_head=8
            d_model=1024, d_inner=512, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, tar_seq, return_attns=False):
        dec_slf_attn_list = []
        enc_output = src_seq
        dec_output = tar_seq
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(enc_output, dec_output)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
        if return_attns:
            return dec_output, dec_slf_attn_list
        return dec_output,

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # set_trace()
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, dec_input, non_pad_mask=None, slf_attn_mask=None):
        dec_output, dec_slf_attn = self.enc_attn(
            dec_input, enc_input, enc_input, mask=slf_attn_mask)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qqs = nn.Linear(d_model, self.n_head * self.d_k)
        # self.w_qqs = nn.Linear(51, 52)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qqs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, qq, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_qq, _ = qq.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = qq
        # set_trace()
        qq = self.w_qqs(qq).view(sz_b, len_qq, n_head, d_k)

        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        qq = qq.permute(2, 0, 1, 3).contiguous().view(-1, len_qq, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        # output, attn = self.attention(qq, k, v, mask=mask)
        output, attn = self.attention(qq, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_qq, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_qq, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        # set_trace()

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn