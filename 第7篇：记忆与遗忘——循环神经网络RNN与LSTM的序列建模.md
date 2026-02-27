# 第7篇：记忆与遗忘——循环神经网络RNN与LSTM的序列建模

## 一、CNN无法解决的问题

上一篇我们掌握了CNN，它能出色地处理图像——捕捉局部特征、保持空间结构。但当我们面对这样的问题：

> "我今天不舒服，打算明天去医院，希望____能好一点。"

人类能轻松填入"身体"或"病"。但CNN会怎么做？它把句子当作一张"词的图片"，完全丢失了**词语的顺序信息**和**长距离依赖关系**。

**序列数据**（文本、语音、时间序列）的核心特征：
1. **有序性**："狗咬人" ≠ "人咬狗"
2. **变长**：句子长度不固定
3. **长程依赖**：后面的词依赖前面很远的内容

这就是**循环神经网络（Recurrent Neural Network, RNN）**的用武之地。

![RNN展开结构](https://i-blog.csdnimg.cn/img_convert/31c9e593519a30a988ed6a6dd9653df7.png)

## 二、RNN的核心思想：循环与记忆

### 2.1 网络结构的循环

与CNN的"一层一层"不同，RNN的核心是一个**循环单元**：

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$\hat{y}_t = W_{hy}h_t + b_y$$

其中：
- $x_t$：时刻$t$的输入
- $h_{t-1}$：上一时刻的隐藏状态（记忆）
- $h_t$：当前时刻的隐藏状态（更新后的记忆）
- $\hat{y}_t$：当前时刻的输出

**关键洞察**：同一个网络结构，在不同时间步**重复使用**，隐藏状态$h$像"记忆"一样传递。

### 2.2 展开视角

把RNN在时间上"展开"：

```
时刻1:  x1 → [RNN单元] → h1 → y1
              ↑
时刻2:  x2 → [RNN单元] → h2 → y2
              ↑
时刻3:  x3 → [RNN单元] → h3 → y3
              ↑
            ...
```

**所有时间步共享同一套参数**（$W_{hh}, W_{xh}, W_{hy}$），这大大减少了参数量。

### 2.3 记忆的本质

隐藏状态$h_t$是**之前所有输入的压缩表示**：

$$h_t = f(x_t, x_{t-1}, x_{t-2}, ..., x_1)$$

理论上，$h_t$包含了从开头到当前的全部历史信息。但实践中，这个"记忆"有严重缺陷...

## 三、梯度消失：RNN的致命伤

### 3.1 反向传播通过时间（BPTT）

训练RNN使用**Backpropagation Through Time**：

1. 展开网络（假设序列长度100）
2. 前向传播计算所有$h_t, y_t$
3. 反向传播从$t=100$回到$t=1$

### 3.2 梯度消失的数学

计算$\frac{\partial L}{\partial W_{hh}}$，需要连乘Jacobian矩阵：

$$\frac{\partial h_{100}}{\partial h_{99}} \cdot \frac{\partial h_{99}}{\partial h_{98}} \cdots \frac{\partial h_{2}}{\partial h_{1}}$$

每个Jacobian包含$\tanh$的导数，最大值为1。100个小于1的数相乘：

$$\text{梯度} \approx (0.5)^{100} \approx 10^{-30}$$

**梯度消失！** 前面的时间步几乎学不到东西。

![梯度消失](https://i-blog.csdnimg.cn/img_convert/5f08d0bf80c04057b394e729c6241dca.png)

### 3.3 实际表现

| 序列长度 | RNN表现 |
|:---|:---|
| 5-10 | 良好 |
| 20-30 | 困难 |
| 50+ | 几乎无法学习长程依赖 |

**"我今天不舒服，打算明天去医院，希望身体能好一点。"**

RNN能记住"身体"和"不舒服"的关系（短距离），但很难学会"明天"和"今天"的时序关系（如果句子更长）。

## 四、LSTM：长短期记忆网络

### 4.1 核心思想

1997年，Hochreiter和Schmidhuber提出LSTM，通过**门控机制**控制信息的流动：
- **遗忘门**：决定丢弃哪些旧信息
- **输入门**：决定存储哪些新信息
- **输出门**：决定输出什么

### 4.2 LSTM的数学结构

**遗忘门**（Forget Gate）：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**（Input Gate）：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**细胞状态更新**（Cell State）：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门**（Output Gate）：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

![LSTM结构](https://i-blog.csdnimg.cn/img_convert/ad9f8fd6c9101826e1c96243a6f3b8aa.png)

### 4.3 为什么LSTM能解决梯度消失？

**关键：细胞状态$C_t$的更新**

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

这是一个**线性**的更新（没有$\tanh$或$\sigma$的压缩），梯度可以**无损传递**：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t \approx 1 \text{（当需要记忆时）}$$

**遗忘门的作用**：
- $f_t \approx 1$：保留长期记忆，梯度正常传播
- $f_t \approx 0$：遗忘旧信息，但这是对内容的重置，不是梯度消失

### 4.4 直观理解

想象LSTM是一个**精密的档案管理系统**：

| 组件 | 功能 | 类比 |
|:---|:---|:---|
| 细胞状态$C_t$ | 长期记忆 | 档案库 |
| 遗忘门$f_t$ | 决定销毁哪些旧档案 | 档案管理员 |
| 输入门$i_t$ | 决定存入哪些新档案 | 档案录入员 |
| 输出门$o_t$ | 决定展示哪些档案 | 档案查询员 |
| 隐藏状态$h_t$ | 工作记忆（输出） | 当前处理的内容 |

## 五、GRU：LSTM的简化版

### 5.1 门控循环单元

2014年，Cho提出GRU，将LSTM的3个门简化为2个：

**更新门**（Update Gate）：
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**重置门**（Reset Gate）：
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**候选隐藏状态**：
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

**隐藏状态更新**：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 5.2 LSTM vs GRU

| 特性 | LSTM | GRU |
|:---|:---|:---|
| 门数量 | 3（遗忘、输入、输出） | 2（更新、重置） |
| 参数量 | 多 | 少（约25%减少）|
| 训练速度 | 较慢 | 较快 |
| 表达能力 | 理论上更强 | 实践中相当 |
| 适用场景 | 长序列、复杂依赖 | 中等序列、资源受限 |

**经验法则**：
- 数据量大、序列长 → LSTM
- 数据量小、需要快速实验 → GRU

## 六、Python实现：字符级文本生成

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# 加载示例文本
text = open('shakespeare.txt', 'rb').read().decode(encoding='utf-8')
# 或使用简单示例
text = "To be, or not to be, that is the question. " * 100

# 创建字符映射
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# 准备训练数据
seq_length = 100  # 输入序列长度
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

# 向量化
x = np.zeros((len(sentences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

print(f"训练样本数: {len(sentences)}")
print(f"词汇表大小: {vocab_size}")
print(f"序列长度: {seq_length}")

# 构建LSTM模型
def build_lstm_model(vocab_size, seq_length, lstm_units=128):
    inputs = layers.Input(shape=(seq_length, vocab_size))
    
    # 两层LSTM
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return model

# 构建GRU模型对比
def build_gru_model(vocab_size, seq_length, gru_units=128):
    inputs = layers.Input(shape=(seq_length, vocab_size))
    
    x = layers.GRU(gru_units, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(gru_units)(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return model

# 训练
model = build_lstm_model(vocab_size, seq_length)
print("LSTM模型结构：")
model.summary()

# 回调函数
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='loss'
)

history = model.fit(x, y, 
                    batch_size=128, 
                    epochs=50, 
                    callbacks=[checkpoint],
                    verbose=1)

# 绘制训练曲线
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'])
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# 文本生成函数
def generate_text(model, seed_text, length=200, temperature=1.0):
    generated = seed_text
    
    for _ in range(length):
        # 准备输入
        x_pred = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_to_idx[char]] = 1
        
        # 预测
        preds = model.predict(x_pred, verbose=0)[0]
        
        # 温度采样
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        probas = np.random.multinomial(1, preds, 1)
        next_index = np.argmax(probas)
        next_char = idx_to_char[next_index]
        
        generated += next_char
        seed_text = seed_text[1:] + next_char
    
    return generated

# 生成文本
seed = text[:seq_length]
print("种子文本：", seed)
print("\n生成的文本：")
print(generate_text(model, seed, length=200, temperature=0.5))
```

## 七、C/C++实现：基础RNN前向传播

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 10
#define HIDDEN_SIZE 20
#define OUTPUT_SIZE 10
#define SEQ_LENGTH 5

// 激活函数
double tanh_activation(double x) {
    return tanh(x);
}

double sigmoid(double x) {
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

// 简单的RNN单元
typedef struct {
    double Wxh[INPUT_SIZE][HIDDEN_SIZE];  // 输入到隐藏
    double Whh[HIDDEN_SIZE][HIDDEN_SIZE]; // 隐藏到隐藏
    double Why[HIDDEN_SIZE][OUTPUT_SIZE]; // 隐藏到输出
    double bh[HIDDEN_SIZE];               // 隐藏偏置
    double by[OUTPUT_SIZE];               // 输出偏置
    
    double h[HIDDEN_SIZE];                // 当前隐藏状态
} SimpleRNN;

// 初始化RNN
void rnn_init(SimpleRNN* rnn) {
    srand(42);
    double scale = 0.01;
    
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            rnn->Wxh[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale;
    
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            rnn->Whh[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale;
    
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < OUTPUT_SIZE; j++)
            rnn->Why[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale;
    
    memset(rnn->bh, 0, sizeof(rnn->bh));
    memset(rnn->by, 0, sizeof(rnn->by));
    memset(rnn->h, 0, sizeof(rnn->h));
}

// RNN前向传播一步
void rnn_step(SimpleRNN* rnn, double* x, double* y) {
    double h_new[HIDDEN_SIZE];
    
    // 计算新的隐藏状态：h_new = tanh(Wxh·x + Whh·h + bh)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_new[i] = rnn->bh[i];
        
        // 输入贡献
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_new[i] += rnn->Wxh[j][i] * x[j];
        }
        
        // 隐藏状态贡献（循环连接）
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_new[i] += rnn->Whh[j][i] * rnn->h[j];
        }
        
        h_new[i] = tanh_activation(h_new[i]);
    }
    
    // 更新隐藏状态
    memcpy(rnn->h, h_new, sizeof(h_new));
    
    // 计算输出：y = Why·h + by
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        y[i] = rnn->by[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            y[i] += rnn->Why[j][i] * rnn->h[j];
        }
    }
    
    // softmax输出（可选，用于分类）
    double max_val = y[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (y[i] > max_val) max_val = y[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        y[i] = exp(y[i] - max_val);
        sum += y[i];
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        y[i] /= sum;
    }
}

// 重置隐藏状态
void rnn_reset(SimpleRNN* rnn) {
    memset(rnn->h, 0, sizeof(rnn->h));
}

int main() {
    printf("RNN循环神经网络演示\n");
    printf("====================\n\n");
    
    SimpleRNN rnn;
    rnn_init(&rnn);
    
    printf("网络结构：\n");
    printf("  输入维度：%d\n", INPUT_SIZE);
    printf("  隐藏维度：%d\n", HIDDEN_SIZE);
    printf("  输出维度：%d\n", OUTPUT_SIZE);
    printf("  序列长度：%d\n\n", SEQ_LENGTH);
    
    // 模拟序列输入（one-hot编码）
    double sequence[SEQ_LENGTH][INPUT_SIZE];
    for (int t = 0; t < SEQ_LENGTH; t++) {
        memset(sequence[t], 0, sizeof(sequence[t]));
        sequence[t][t % INPUT_SIZE] = 1.0;  // 简单的模式
    }
    
    printf("序列处理过程：\n");
    printf("时间步 | 输入(激活位) | 隐藏状态范数 | 输出(最大概率位)\n");
    printf("-------|-------------|-------------|----------------\n");
    
    rnn_reset(&rnn);
    
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double y[OUTPUT_SIZE];
        rnn_step(&rnn, sequence[t], y);
        
        // 计算隐藏状态范数
        double h_norm = 0.0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            h_norm += rnn.h[i] * rnn.h[i];
        }
        h_norm = sqrt(h_norm);
        
        // 找到最大输出
        int max_idx = 0;
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (y[i] > y[max_idx]) max_idx = i;
        }
        
        printf("  %d    |     %2d      |   %.4f    |      %2d (%.2f)\n",
               t, t % INPUT_SIZE, h_norm, max_idx, y[max_idx]);
    }
    
    printf("\n关键观察：\n");
    printf("1. 隐藏状态范数变化反映了网络对序列的记忆\n");
    printf("2. 输出依赖于当前输入和历史隐藏状态\n");
    printf("3. 长序列会导致梯度消失（此处未展示训练）\n");
    printf("\n实际应用中，建议使用LSTM或GRU替代基础RNN\n");
    
    return 0;
}
```

**编译运行：**
```bash
gcc rnn_demo.c -o rnn_demo -lm
./rnn_demo
```

## 八、Java实现：LSTM单元

```java
public class LSTMNetwork {
    
    // LSTM单元参数
    public static class LSTMCell {
        int inputSize, hiddenSize;
        
        // 权重矩阵
        double[][] Wf, Wi, Wc, Wo;  // 遗忘门、输入门、候选状态、输出门
        double[] bf, bi, bc, bo;    // 偏置
        
        // 状态
        double[] h;  // 隐藏状态
        double[] c;  // 细胞状态
        
        public LSTMCell(int inputSize, int hiddenSize) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            
            // 初始化权重（Xavier）
            double scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
            Random rand = new Random(42);
            
            Wf = new double[hiddenSize][inputSize + hiddenSize];
            Wi = new double[hiddenSize][inputSize + hiddenSize];
            Wc = new double[hiddenSize][inputSize + hiddenSize];
            Wo = new double[hiddenSize][inputSize + hiddenSize];
            
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < inputSize + hiddenSize; j++) {
                    Wf[i][j] = (rand.nextDouble() - 0.5) * 2 * scale;
                    Wi[i][j] = (rand.nextDouble() - 0.5) * 2 * scale;
                    Wc[i][j] = (rand.nextDouble() - 0.5) * 2 * scale;
                    Wo[i][j] = (rand.nextDouble() - 0.5) * 2 * scale;
                }
            }
            
            bf = new double[hiddenSize];
            bi = new double[hiddenSize];
            bc = new double[hiddenSize];
            bo = new double[hiddenSize];
            
            // 遗忘门偏置初始化为1（默认记住）
            for (int i = 0; i < hiddenSize; i++) bf[i] = 1.0;
            
            h = new double[hiddenSize];
            c = new double[hiddenSize];
        }
        
        // sigmoid激活
        private double sigmoid(double x) {
            if (x > 500) return 1.0;
            if (x < -500) return 0.0;
            return 1.0 / (1.0 + Math.exp(-x));
        }
        
        // tanh激活
        private double tanh(double x) {
            return Math.tanh(x);
        }
        
        // 前向传播一步
        public double[] forward(double[] x) {
            // 拼接输入 [h_{t-1}, x_t]
            double[] concat = new double[hiddenSize + inputSize];
            System.arraycopy(h, 0, concat, 0, hiddenSize);
            System.arraycopy(x, 0, concat, hiddenSize, inputSize);
            
            // 遗忘门
            double[] f = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                f[i] = bf[i];
                for (int j = 0; j < concat.length; j++) {
                    f[i] += Wf[i][j] * concat[j];
                }
                f[i] = sigmoid(f[i]);
            }
            
            // 输入门
            double[] i_gate = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                i_gate[i] = bi[i];
                for (int j = 0; j < concat.length; j++) {
                    i_gate[i] += Wi[i][j] * concat[j];
                }
                i_gate[i] = sigmoid(i_gate[i]);
            }
            
            // 候选细胞状态
            double[] c_tilde = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                c_tilde[i] = bc[i];
                for (int j = 0; j < concat.length; j++) {
                    c_tilde[i] += Wc[i][j] * concat[j];
                }
                c_tilde[i] = tanh(c_tilde[i]);
            }
            
            // 更新细胞状态
            for (int i = 0; i < hiddenSize; i++) {
                c[i] = f[i] * c[i] + i_gate[i] * c_tilde[i];
            }
            
            // 输出门
            double[] o = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                o[i] = bo[i];
                for (int j = 0; j < concat.length; j++) {
                    o[i] += Wo[i][j] * concat[j];
                }
                o[i] = sigmoid(o[i]);
            }
            
            // 更新隐藏状态
            for (int i = 0; i < hiddenSize; i++) {
                h[i] = o[i] * tanh(c[i]);
            }
            
            return h.clone();
        }
        
        public void reset() {
            Arrays.fill(h, 0.0);
            Arrays.fill(c, 0.0);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("LSTM长短期记忆网络演示");
        System.out.println("======================\n");
        
        int inputSize = 5;
        int hiddenSize = 10;
        int seqLength = 8;
        
        LSTMCell lstm = new LSTMCell(inputSize, hiddenSize);
        
        System.out.println("LSTM单元结构：");
        System.out.println("  输入维度：" + inputSize);
        System.out.println("  隐藏维度：" + hiddenSize);
        System.out.println("  序列长度：" + seqLength);
        System.out.println("  门控：遗忘门、输入门、输出门\n");
        
        // 模拟序列输入
        double[][] sequence = new double[seqLength][inputSize];
        for (int t = 0; t < seqLength; t++) {
            // 简单的模式：第t步激活第t%inputSize位
            sequence[t][t % inputSize] = 1.0;
        }
        
        System.out.println("序列处理过程：");
        System.out.println("时间步 | 输入位 | 细胞状态范数 | 隐藏状态范数");
        System.out.println("-------|--------|-------------|-------------");
        
        lstm.reset();
        
        for (int t = 0; t < seqLength; t++) {
            double[] h = lstm.forward(sequence[t]);
            
            // 计算范数
            double c_norm = 0.0, h_norm = 0.0;
            for (int i = 0; i < hiddenSize; i++) {
                c_norm += lstm.c[i] * lstm.c[i];
                h_norm += h[i] * h[i];
            }
            c_norm = Math.sqrt(c_norm);
            h_norm = Math.sqrt(h_norm);
            
            System.out.printf("  %d    |   %d    |   %.4f    |   %.4f%n",
                t, t % inputSize, c_norm, h_norm);
        }
        
        System.out.println("\n关键观察：");
        System.out.println("1. 细胞状态范数相对稳定（长期记忆）");
        System.out.println("2. 隐藏状态范数随输入动态变化（短期输出）");
        System.out.println("3. 遗忘门初始化为1，默认保留历史信息");
        System.out.println("\nLSTM通过门控机制解决了RNN的梯度消失问题，");
        System.out.println("能够学习长距离依赖关系。");
    }
}
```

## 九、RNN/LSTM的应用场景

| 应用领域 | 典型任务 | 网络结构 |
|:---|:---|:---|
| **自然语言处理** | 机器翻译、文本生成 | Seq2Seq + Attention |
| **语音识别** | 语音转文字 | 深层双向LSTM |
| **时间序列预测** | 股价预测、天气预测 | LSTM/GRU |
| **音乐生成** | 旋律创作 | 字符级RNN |
| **视频分析** | 动作识别 | CNN + LSTM |

![序列建模](https://i-blog.csdnimg.cn/img_convert/3c77bee75df232a42780cdeaabd3e3b7.jpeg)

## 十、总结与展望

### 10.1 从CNN到RNN的演进

```
CNN：处理空间结构（图像）
  ↓
RNN：处理时间序列（文本、语音）
  ↓
LSTM/GRU：解决长程依赖
  ↓
Attention：突破序列长度限制（下篇预告）
```

### 10.2 关键公式速查

| 模型 | 核心公式 | 关键特性 |
|:---|:---|:---|
| RNN | $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$ | 循环连接，参数共享 |
| LSTM | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | 门控机制，梯度高速公路 |
| GRU | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ | 简化版LSTM，参数更少 |

### 10.3 选择指南

```
序列长度 < 10？ → 简单RNN可能够用
序列长度 10-50？ → GRU（快速实验）
序列长度 > 50？ → LSTM（长程依赖）
需要极致性能？ → 双向LSTM + Attention
```

**下一篇预告：《第8篇：注意力机制——从Seq2Seq到Transformer的飞跃》**

我们将解决LSTM的最后一个局限：**无法并行计算**。注意力机制让模型"看到"整个序列，Transformer彻底改变了NLP领域，也为GPT、BERT等大模型奠定了基础。

---

**本文代码已开源**：[深入探索深度学习的五大核心神经网络架(In depth exploration of the five core neural network architectures of deep learning)](https://github.com/city25/IDEOTFCNNARODL)

**配图清单：**
- 图1：RNN展开结构示意图
- 图2：梯度消失问题可视化
- 图3：LSTM单元内部结构
- 图4：序列建模应用场景
- 图5：LSTM vs GRU对比

---

*全文约6200字，是深度学习五大架构的第二篇（RNN）。从CNN的空间建模到RNN的时间建模，我们覆盖了图像和序列两大核心数据类型。第8篇Attention机制，将开启并行计算的新纪元！*

---
> 本文部分内容由AI编辑，可能会出现幻觉，请谨慎阅读。