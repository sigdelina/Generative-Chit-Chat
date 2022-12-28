import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, hid_dim, device):
        super().__init__()

        # сверточные слои
        self.conv1 = nn.Conv1d(in_channels=hid_dim,
                               out_channels=hid_dim,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv1d(in_channels=hid_dim,
                               out_channels=hid_dim,
                               kernel_size=5,
                               padding=2)

        self.batch_norm = nn.BatchNorm1d(hid_dim)
        self.activ = nn.ReLU()
        self.scale = torch.sqrt(torch.FloatTensor([0.7])).to(device)
        self.device = device

    def forward(self, to_conv_perm):
        # сверточные слои
        conv1 = self.conv1(to_conv_perm)  # [batch, hid_dim, len_sent]
        conv1 = self.batch_norm(conv1)
        # при gelu, out_chanels должен задаваться как 2 * hid_dim, так как в ней зашито деление на 2
        conv1 = self.activ(conv1)  # [batch, hid_dim, len_sent]
        conv_after_resid = (to_conv_perm + conv1) * self.scale  # [batch, hid_dim, len_sent]

        conv2 = self.conv2(conv_after_resid)  # [batch, hid_dim, len_sent]
        conv2 = self.batch_norm(conv2)
        conv2 = self.activ(conv2)  # [batch, hid_dim, len_sent]
        conv_after_2resid = (conv_after_resid + conv2) * self.scale  # [batch, hid_dim, len_sent]

        # смена размерности для линейного слоя
        to_out_lin = conv_after_2resid.permute(0, 2, 1)  # [batch, len_sent, hid_dim]

        return to_out_lin


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, kernel_s, device, max_length=64):
        super().__init__()

        # инициализируем эмбеддинги для токенов и их позиций (от первого токена до последнего)
        self.position_embedding = nn.Embedding(max_length, emb_dim)
        self.token_embedding = nn.Embedding(input_dim, emb_dim)

        # линейные слои
        self.linear_toblock = nn.Linear(emb_dim, hid_dim)
        self.linear_outblock = nn.Linear(hid_dim, emb_dim)

        # сверточные слои
        self.scale = torch.sqrt(torch.FloatTensor([0.7])).to(device)
        self.conv_b = ConvBlock(hid_dim, device)
        self.device = device

    def forward(self, b_elem):
        batch_size = b_elem.shape[0]  # размер батча
        src_len = b_elem.shape[1]  # длина последовательности

        # генерация матрицы тензора для позиций слов: [batch, len_sent]
        posintion_t = torch.stack([torch.arange(0, src_len) for pos in range(batch_size)]).to(self.device)
        # получаем эмбеддинги: [batch, len_sent, embed_dim]
        tok_emb = self.token_embedding(b_elem.to(self.device))

        pos_emb = self.position_embedding(posintion_t)

        # elemenwise sum: [batch, len_sent, embed_dim]
        elem_wise = tok_emb + pos_emb  # тут еще можно добавить дропаут

        # линейные слой: [batch, len_sent, hid_dim]
        to_block = self.linear_toblock(elem_wise)

        # входная размерность в conv - (N,C,L), где L - длина последоваетльности, С - число каналов
        # [batch, hid_dim, len_sent]
        to_conv_perm = to_block.permute(0, 2, 1)

        # # сверточные слои
        to_out_lin = self.conv_b(to_conv_perm)

        # линейный слой
        out_lin = self.linear_outblock(to_out_lin)  # [batch, len_sent, emb_dim]

        # residual connection
        res_con_out = (out_lin + elem_wise) * self.scale  # [batch, len_sent, emb_dim]

        return out_lin, res_con_out


class Attention(nn.Module):
    def __init__(self, emb_size, hid_size, device):
        super(Attention, self).__init__()
        self.emb2hid = nn.Linear(emb_size, hid_size)
        self.hid2emb = nn.Linear(hid_size, emb_size)
        self.scale = torch.sqrt(torch.FloatTensor([0.7])).to(device)

    def forward(self, dec_conved, embedd, en_conved, en_combined):
        dec_conved = dec_conved.permute(0, 2, 1)
        dec_conved_emb = self.hid2emb(dec_conved)

        Q = (dec_conved_emb + embedd) * self.scale
        energy = torch.matmul(Q, en_conved.permute(0, 2, 1))
        a = F.softmax(energy, dim=2)

        context = torch.matmul(a, en_combined)
        context = self.emb2hid(context)
        conved = (context + dec_conved) * self.scale

        return a, conved.permute(0, 2, 1)


class ConvBlock_Decod(nn.Module):
    def __init__(self, emb_dim, hid_dim, pad_param1, pad_param2, device):
        super().__init__()

        self.pad_param1 = pad_param1
        self.pad_param2 = pad_param2

        # сверточные слои
        self.conv1 = nn.Conv1d(in_channels=hid_dim,
                               out_channels=hid_dim,
                               kernel_size=3 + 2,
                               padding=self.pad_param1)

        self.conv2 = nn.Conv1d(in_channels=hid_dim,
                               out_channels=hid_dim,
                               kernel_size=5,
                               padding=self.pad_param2)

        self.batch_norm = nn.BatchNorm1d(hid_dim)

        self.attent = Attention(emb_dim, hid_dim, device)

        self.scale = torch.sqrt(torch.FloatTensor([0.7])).to(device)

        self.activ = nn.ReLU()
        self.device = device

    def forward(self, padded_seq, to_conv_perm, elem_wise, enc_output, enc_res_output):
        # сверточные слои
        conv1 = self.conv1(padded_seq)  # [batch, hid_dim, len_sent]
        conv1 = self.batch_norm(conv1)
        conv1 = self.activ(conv1)  # [batch, hid_dim, len_sent]

        attentioned, out_conv = self.attent(conv1, elem_wise, enc_output, enc_res_output)

        conv_after_resid = (to_conv_perm + out_conv) * self.scale  # [batch, hid_dim, len_sent]

        conv2 = self.conv2(conv_after_resid)  # [batch, hid_dim, len_sent]
        conv2 = self.batch_norm(conv2)
        conv2 = self.activ(conv2)  # [batch, hid_dim, len_sent]

        attentioned2, out_conv2 = self.attent(conv2, elem_wise, enc_output, enc_res_output)
        conv_after_2resid = (conv_after_resid + out_conv2) * self.scale  # [batch, hid_dim, len_sent]
        return conv_after_2resid, attentioned2


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, kernel_s, device, pad_idx=0, max_length=64):
        super().__init__()

        self.pad_idx = pad_idx
        self.pad_param1 = 1
        self.pad_param2 = 2
        self.emb_dim = emb_dim

        self.kern_s = kernel_s

        # инициализируем эмбеддинги для токенов и их позиций (от первого токена до последнего)
        self.position_embedding = nn.Embedding(max_length, emb_dim)
        self.token_embedding = nn.Embedding(output_dim, emb_dim)

        # линейные слои
        self.linear_toblock = nn.Linear(emb_dim, hid_dim)
        self.linear_outblock = nn.Linear(hid_dim, emb_dim)
        self.liner_out = nn.Linear(emb_dim, output_dim)

        self.conv_blocks = ConvBlock_Decod(self.emb_dim, hid_dim, self.pad_param1, self.pad_param2, device)

        self.dropout = nn.Dropout(0.05)
        self.device = device

    def forward(self, b_elem, enc_output, enc_res_output):
        batch_size = b_elem.shape[0]  # размер батча
        src_len = b_elem.shape[1]  # длина последовательности

        # генерация матрицы тензора для позиций слов: [batch, len_sent]
        posintion_t = torch.stack([torch.arange(0, src_len) for pos in range(batch_size)]).to(self.device)

        # получаем эмбеддинги: [batch, len_sent, embed_dim]
        tok_emb = self.token_embedding(b_elem.to(self.device))
        pos_emb = self.position_embedding(posintion_t)

        # elemenwise sum: [batch, len_sent, embed_dim]
        elem_wise = tok_emb + pos_emb  # тут еще можно добавить дропаут

        # линейные слой: [batch, len_sent, hid_dim]
        to_block = self.linear_toblock(elem_wise)

        # входная размерность в conv - (N,C,L), где L - длина последоваетльности, С - число каналов
        # [batch, hid_dim, len_sent]
        to_conv_perm = to_block.permute(0, 2, 1)

        # делаем паддинги
        padding = torch.zeros(to_block.shape[0], to_block.shape[2], self.pad_param1 + 1).fill_(self.pad_idx).to(
            self.device)
        # срезаем маркер конца предложения
        padded_seq = torch.cat((padding, to_conv_perm), dim=2)

        # # сверточные слои
        conv_after_2resid, attentioned2 = self.conv_blocks(padded_seq, to_conv_perm, elem_wise, enc_output,
                                                           enc_res_output)

        # смена размерности для линейного слоя
        to_out_lin = conv_after_2resid.permute(0, 2, 1)  # [batch, len_sent, hid_dim]

        # линейный слой
        out_lin = self.linear_outblock(to_out_lin)  # [batch, len_sent, emb_dim]

        drop = self.dropout(out_lin)
        fn = self.liner_out(drop)
        # # residual connection
        # res_con_out = out_lin + elem_wise  # [batch, len_sent, emb_dim]

        return fn, attentioned2


class CNN_Seq2Seq(nn.Module):

    def __init__(self, input_dim, output_dim, enc_emb_dim, dec_emb_dim, hid_dim, kernel_s, device, pad_idx=0,
                 max_length=64):
        super().__init__()
        self.encoder = Encoder(input_dim, enc_emb_dim, hid_dim, kernel_s, device)
        self.decoder = Decoder(output_dim, dec_emb_dim, hid_dim, kernel_s, device)

    def forward(self, sourse, target):
        sourse_output, sourse_output_resid = self.encoder(x)

        target_output, target_atten = self.decoder(target, sourse_output, sourse_output_resid)

        return target_output, target_atten

