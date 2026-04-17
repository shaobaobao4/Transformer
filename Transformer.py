import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ==========================================
# 1. 加性注意力机制 (Additive Attention)
# ==========================================
class MultiHeadAdditiveAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAdditiveAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 初始的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 加性注意力特有的权重 W_q_att, W_k_att 和 v
        self.W_q_att = nn.Linear(self.d_k, self.d_k)
        self.W_k_att = nn.Linear(self.d_k, self.d_k)
        self.v_att = nn.Linear(self.d_k, 1, bias=False)

        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性映射并分头 -> [batch, num_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 加性注意力的打分计算：v^T * tanh(W_q Q + W_k K)
        # 扩展维度以支持广播相加
        Q_att = self.W_q_att(Q).unsqueeze(3)  # [batch, num_heads, seq_len_q, 1, d_k]
        K_att = self.W_k_att(K).unsqueeze(2)  # [batch, num_heads, 1, seq_len_k, d_k]

        energy = torch.tanh(Q_att + K_att)  # [batch, num_heads, seq_len_q, seq_len_k, d_k]
        scores = self.v_att(energy).squeeze(-1)  # [batch, num_heads, seq_len_q, seq_len_k]

        if mask is not None:
            mask = mask.unsqueeze(1)  # 匹配多头维度
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 3. 注意力权重与 V 相乘 -> [batch, num_heads, seq_len_q, d_k]
        context = torch.matmul(attn, V)

        # 4. 拼接多头并输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(context)
        return output


# ==========================================
# 2. Transformer 其他模块
# ==========================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAdditiveAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.layernorm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.layernorm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAdditiveAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAdditiveAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.layernorm1(x + self.dropout(self.self_attn(x, x, x, trg_mask)))
        x = self.layernorm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.layernorm3(x + self.dropout(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.trg_embed = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, trg, pad_idx):
        src_mask = (src != pad_idx).unsqueeze(1)
        if trg is not None:
            trg_pad_mask = (trg != pad_idx).unsqueeze(1)
            trg_sub_mask = torch.tril(torch.ones((trg.size(1), trg.size(1)), device=trg.device)).bool()
            trg_mask = trg_pad_mask & trg_sub_mask
            return src_mask, trg_mask
        return src_mask, None

    def forward(self, src, trg, pad_idx):
        src_mask, trg_mask = self.generate_mask(src, trg, pad_idx)
        enc_out = self.dropout(self.pos_encode(self.src_embed(src)))
        for layer in self.encoder_layers: enc_out = layer(enc_out, src_mask)

        dec_out = self.dropout(self.pos_encode(self.trg_embed(trg)))
        for layer in self.decoder_layers: dec_out = layer(dec_out, enc_out, src_mask, trg_mask)

        return self.fc_out(dec_out)


# ==========================================
# 3. 数据预处理
# ==========================================
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class TranslationDataset(Dataset):
    def __init__(self, filepath, src_vocab=None, trg_vocab=None):
        self.src_data, self.trg_data = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.src_data.append(parts[0].split())
                    self.trg_data.append(parts[1].split())

        if src_vocab is None:
            self.src_vocab, self.src_word2idx = self.build_vocab(self.src_data)
            self.trg_vocab, self.trg_word2idx = self.build_vocab(self.trg_data)
            self.trg_idx2word = {i: w for w, i in self.trg_word2idx.items()}
        else:
            self.src_word2idx = src_vocab
            self.trg_word2idx = trg_vocab
            self.trg_idx2word = {i: w for w, i in self.trg_word2idx.items()}

    def build_vocab(self, data):
        counter = Counter()
        for sentence in data: counter.update(sentence)
        vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + [word for word, _ in counter.most_common(10000)]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        return vocab, word2idx

    def __len__(self):
        return len(self.src_data)

    def numericalize(self, sentence, word2idx):
        return [word2idx.get(w, word2idx[UNK_TOKEN]) for w in sentence]

    def __getitem__(self, idx):
        src_seq = [self.src_word2idx[SOS_TOKEN]] + self.numericalize(self.src_data[idx], self.src_word2idx) + [
            self.src_word2idx[EOS_TOKEN]]
        trg_seq = [self.trg_word2idx[SOS_TOKEN]] + self.numericalize(self.trg_data[idx], self.trg_word2idx) + [
            self.trg_word2idx[EOS_TOKEN]]
        return torch.tensor(src_seq), torch.tensor(trg_seq), self.src_data[idx], self.trg_data[idx]


def collate_fn(batch):
    src_batch, trg_batch, src_text, trg_text = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_padded, trg_padded, src_text, trg_text


# ==========================================
# 4. 推理与翻译逻辑
# ==========================================
def translate_sentence(model, src_tensor, src_pad_idx, trg_sos_idx, trg_eos_idx, max_len=50):
    model.eval()
    device = src_tensor.device

    with torch.no_grad():
        src_mask = (src_tensor != src_pad_idx).unsqueeze(1)
        enc_out = model.dropout(model.pos_encode(model.src_embed(src_tensor)))
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        trg_indexes = [trg_sos_idx]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_pad_mask = (trg_tensor != src_pad_idx).unsqueeze(1)
            trg_sub_mask = torch.tril(torch.ones((trg_tensor.size(1), trg_tensor.size(1)), device=device)).bool()
            trg_mask = trg_pad_mask & trg_sub_mask

            dec_out = model.dropout(model.pos_encode(model.trg_embed(trg_tensor)))
            for layer in model.decoder_layers:
                dec_out = layer(dec_out, enc_out, src_mask, trg_mask)

            output = model.fc_out(dec_out)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == trg_eos_idx:
                break
    return trg_indexes


# ==========================================
# 5. 主训练与保存逻辑
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    train_dataset = TranslationDataset("D:\eng-fra_train_data(1)等2项文件\eng-fra_train_data(1).txt")
    test_dataset = TranslationDataset("D:\eng-fra_train_data(1)等2项文件\eng-fra_test_data(1).txt", train_dataset.src_word2idx,
                                      train_dataset.trg_word2idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    pad_idx = train_dataset.src_word2idx[PAD_TOKEN]
    trg_sos_idx = train_dataset.trg_word2idx[SOS_TOKEN]
    trg_eos_idx = train_dataset.trg_word2idx[EOS_TOKEN]

    # 2. 初始化加性注意力 Transformer
    model = Transformer(
        src_vocab_size=len(train_dataset.src_word2idx),
        trg_vocab_size=len(train_dataset.trg_word2idx),
        d_model=128, num_heads=4, num_layers=2, d_ff=256
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 3. 训练循环 (这里演示训练5轮)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, trg, _, _ in train_loader:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            optimizer.zero_grad()
            predictions = model(src, trg_input, pad_idx)

            loss = criterion(predictions.reshape(-1, predictions.shape[-1]), trg_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # 4. 翻译测试集并保存到 E 盘
    save_path = "E:\\translation_results.txt"
    print(f"训练完成，正在翻译测试集并保存至: {save_path} ...")

    # 确保保存目录存在(如果指定了子目录)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("英文原文\t真实法文\t模型预测法文\n")
        f.write("=" * 60 + "\n")
        for idx in range(len(test_dataset)):
            src_tensor, _, src_text, trg_text = test_dataset[idx]
            src_tensor = src_tensor.unsqueeze(0).to(device)

            # 推理得到预测的 Token ID
            pred_indexes = translate_sentence(model, src_tensor, pad_idx, trg_sos_idx, trg_eos_idx)

            # ID 转为 单词
            pred_words = [test_dataset.trg_idx2word[i] for i in pred_indexes]
            # 去除首尾的 <sos> 和 <eos>
            pred_words = [w for w in pred_words if w not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]

            # 格式化输出写入E盘文件
            src_str = " ".join(src_text)
            trg_str = " ".join(trg_text)
            pred_str = " ".join(pred_words)
            f.write(f"{src_str}\t{trg_str}\t{pred_str}\n")

            if (idx + 1) % 100 == 0:
                print(f"已翻译 {idx + 1} 条测试数据...")

    print(f"翻译完成！结果已成功保存到 {save_path}")


if __name__ == '__main__':
    main()