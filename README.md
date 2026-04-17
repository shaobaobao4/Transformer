# 基于加性注意力机制的 Transformer 机器翻译模型 (English to French)

本项目实现了一个基于 PyTorch 的自定义 Transformer 模型，专门用于英法 (English-to-French) 机器翻译任务。与标准 Transformer 使用的“缩放点积注意力 (Scaled Dot-Product Attention)”不同，本项目采用 **加性注意力机制 (Additive Attention/Bahdanau Attention)** 来计算注意力权重。

## ✨ 核心特性

* **加性多头注意力 (Multi-Head Additive Attention)**: 自定义注意力层，利用带有可学习权重的单隐藏层前馈网络来计算注意力打分，公式为：$$v^T \tanh(W_q Q + W_k K)$$
* **从零构建 Transformer 组件**: 包含了位置编码 (Positional Encoding)、前馈神经网络 (FFN)、编码器层 (Encoder Layer) 和解码器层 (Decoder Layer) 的完整实现。
* **端到端训练与推理**: 包含自定义的 `Dataset` 与 `DataLoader` 进行词表构建和数据预处理，并实现了自回归的贪心解码 (Greedy Decoding) 用于文本生成。
* **自动化结果导出**: 训练完成后自动对测试集进行翻译，并将原文、真实参考译文以及模型预测的译文对照保存为本地 TXT 文件。

---

## 🛠️ 环境依赖

确保您的环境中已安装以下依赖：

* Python 3.7+
* PyTorch (支持 CUDA 以获得更快的训练速度)
* 基本 Python 标准库: `math`, `os`, `collections`

---

## 📂 模块解析

代码主要由以下五个部分构成：

1. **加性注意力机制 (`MultiHeadAdditiveAttention`)**
   实现了多头加性注意力计算。通过 `W_q_att` 和 `W_k_att` 将 Query 和 Key 投影到相同的维度，相加后经过 `tanh` 激活，最后通过向量 `v_att` 降维得到注意力分数。
2. **Transformer 其他模块**
   包含 `PositionwiseFeedForward`、`PositionalEncoding` 等子模块，并在此基础上构建了 `EncoderLayer`、`DecoderLayer`，最终封装成完整的 `Transformer` 模型。
3. **数据预处理 (`TranslationDataset`)**
   读取制表符分割的双语文本数据，统计词频并截断保留最常见的前 10000 个单词构建词表。提供序列的数字化 (Numericalization) 和 Padding 操作。
4. **推理与翻译逻辑 (`translate_sentence`)**
   将输入的英文句子编码后，在解码器端通过循环一步步生成法文单词，遇到终止符 `<eos>` 或达到最大长度时停止生成。
5. **主训练与保存逻辑 (`main`)**
   配置超参数、优化器 (Adam) 和损失函数 (CrossEntropyLoss，忽略 Padding 标记)。执行 Epoch 循环训练，并在结束后对测试集进行批量推理，结果持久化保存。

---

## 🚀 快速开始

### 1. 准备数据
请确保您的机器翻译数据集（如 `eng-fra.txt`）已放置在正确的路径下。代码默认的数据路径如下，您需要根据实际情况修改 `main()` 函数中的路径：
* **训练集路径**: `D:\eng-fra_train_data(1)等2项文件\eng-fra_train_data(1).txt`
* **测试集路径**: `D:\eng-fra_train_data(1)等2项文件\eng-fra_test_data(1).txt`

> **注意**：数据格式应为每行一句，原文与译文之间用 `\t` (制表符) 分隔。

### 2. 运行模型
直接运行 Python 脚本即可启动训练：
```bash
python your_script_name.py
