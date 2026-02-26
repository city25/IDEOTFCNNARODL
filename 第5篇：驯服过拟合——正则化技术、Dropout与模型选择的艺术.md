# 第5篇：驯服过拟合——正则化技术、Dropout与模型选择的艺术

## 一、一个让人崩溃的场景

经过前面四篇的学习，你终于搭建了一个多层神经网络来解决XOR问题。训练时，损失降到0.0001，准确率100%！你兴奋地把它应用到新数据上——**准确率只有50%，和随机猜一样。**

发生了什么？

**过拟合（Overfitting）**：你的模型"记住"了训练数据，而不是"学会"了规律。就像学生背下了所有例题，但遇到新题就不会了。

这是机器学习中最常见、最棘手的问题之一。本文将带你系统掌握驯服过拟合的五大武器：**L1/L2正则化、Dropout、Early Stopping、数据增强和交叉验证**。

![过拟合与欠拟合](https://kimi-web-img.moonshot.cn/img/analystprep.com/6926b648a2b302b2fc3d76361d4d4ba94a46bae8.jpg)

## 二、理解偏差与方差：诊断你的模型

在解决过拟合之前，我们需要先学会**诊断**。

### 2.1 偏差-方差分解

模型的泛化误差可以分解为三部分：

$$\text{泛化误差} = \text{偏差}^2 + \text{方差} + \text{不可约误差}$$

**偏差（Bias）**：模型对真实关系的错误假设
- 高偏差 → **欠拟合（Underfitting）**
- 表现：训练集和测试集表现都差
- 原因：模型太简单，抓不住规律

**方差（Variance）**：模型对训练数据波动的敏感程度
- 高方差 → **过拟合（Overfitting）**
- 表现：训练集好，测试集差
- 原因：模型太复杂，记住了噪声

![偏差-方差权衡](https://kimi-web-img.moonshot.cn/img/i.sstatic.net/70bd31dfb44ed33190a860b03e150c38da91e600.png)

### 2.2 诊断流程图

```
训练集准确率？
    │
    ├── < 80% → 欠拟合（高偏差）
    │           └── 解决方案：增加模型复杂度、减少正则化
    │
    └── > 95% → 看测试集
                │
                ├── 测试集也好 → 完美！
                │
                └── 测试集差 → 过拟合（高方差）
                              └── 解决方案：本文接下来讲的全部技术
```

## 三、L1/L2正则化：给损失函数加惩罚

### 3.1 核心思想

模型过拟合往往是因为**权重太大**——某些特征被过度重视。正则化通过在损失函数中加入**权重惩罚项**，鼓励模型使用较小的权重。

**L2正则化（Ridge回归）**：
$$L_{\text{reg}} = L_{\text{original}} + \lambda \sum_{i} w_i^2$$

**L1正则化（Lasso回归）**：
$$L_{\text{reg}} = L_{\text{original}} + \lambda \sum_{i} |w_i|$$

其中 $\lambda$ 是正则化强度，控制惩罚力度。

### 3.2 L1 vs L2：逐步理解差异

**从梯度更新的角度：**

L2的梯度：
$$\frac{\partial L_{\text{reg}}}{\partial w_i} = \frac{\partial L}{\partial w_i} + 2\lambda w_i$$

L1的梯度：
$$\frac{\partial L_{\text{reg}}}{\partial w_i} = \frac{\partial L}{\partial w_i} + \lambda \cdot \text{sign}(w_i)$$

**关键差异：**

| 特性 | L2 | L1 |
|:---|:---|:---|
| 惩罚方式 | 权重平方 | 权重绝对值 |
| 大权重 | 惩罚很重 | 线性惩罚 |
| 小权重 | 惩罚很轻 | 恒定惩罚 |
| 效果 | 权重都变小，但**不为零** | 部分权重**变为零** |
| 本质 | 权重收缩 | **特征选择** |

![权重衰减](https://kimi-web-img.moonshot.cn/img/towardsdatascience.com/3f3c2d479b88183b619b11fae2fa9287d56bf930.png)

### 3.3 为什么L1能做特征选择？

想象一个简单场景：两个特征 $x_1, x_2$ 都影响输出，但 $x_1$ 更重要。

**L2**：两个权重都变小，比如 $w_1=0.8, w_2=0.3$
**L1**：由于小权重受到相对更大的惩罚，$w_2$ 可能被压缩到0，只剩 $w_1=1.0$

**几何解释**：L1约束形成菱形区域，更容易与损失函数等高线在坐标轴相交（某些权重为0）。

## 四、Dropout：神经网络的"随机失活"

### 4.1 核心思想

2012年，Hinton提出了Dropout——训练时**随机丢弃**一部分神经元（将其输出设为0）。

**为什么有效？**

**解释1：集成学习视角**
- 每次Dropout相当于训练一个"子网络"
- 最终网络是这些子网络的"平均"
- 类似于训练了多个模型取平均（Bagging）

**解释2：防止共适应**
- 神经元不能依赖其他特定神经元
- 每个神经元必须学会"独立生存"
- 增强了特征的鲁棒性

![Dropout示意图](https://kimi-web-img.moonshot.cn/img/substackcdn.com/342bcb6483e041459475406fae04f146cf328b0c.png)

### 4.2 实现细节

**训练时**：
- 以概率 $p$（如0.5）保留每个神经元
- 被丢弃的神经元的输出设为0
- **重要**：存活神经元的输出要除以 $p$（缩放），保证期望值不变

**测试时**：
- 使用**全部神经元**
- 不需要缩放（训练时已经处理）

### 4.3 Dropout率的设置

| 层类型 | 推荐Dropout率 | 原因 |
|:---|:---:|:---|
| 输入层 | 0.2 | 不要丢弃太多原始信息 |
| 隐藏层 | 0.5 | 标准选择 |
| 输出层 | 0.0 | 通常不Dropout |

**注意**：Dropout只在训练时用，测试时关闭。这会导致训练和测试时分布不同，需要小心处理。

## 五、Early Stopping：及时止损的智慧

### 5.1 核心观察

训练神经网络时，一个普遍现象：
- 训练损失持续下降
- 验证损失先降后升

**验证损失开始上升的那一刻，就是过拟合开始的信号！**

![Early Stopping](https://kimi-web-img.moonshot.cn/img/media.geeksforgeeks.org/1b829c97568aa7288e5c74fd9882d4b5773a0a46.png)

### 5.2 算法实现

```
初始化：best_val_loss = ∞, patience = 10, counter = 0

每轮训练后：
    计算验证损失 val_loss
    
    如果 val_loss < best_val_loss：
        best_val_loss = val_loss
        保存当前模型参数
        counter = 0
    
    否则：
        counter += 1
        如果 counter >= patience：
            停止训练，恢复最佳模型参数
```

**Patience（耐心值）**：允许验证损失不改善的轮数。太小容易早停，太大失去意义。

### 5.3 Early Stopping vs 正则化

| 特性 | Early Stopping | L1/L2/Dropout |
|:---|:---|:---|
| 原理 | 限制训练步数 | 限制模型复杂度 |
| 计算成本 | 低（提前结束） | 高（训练完整） |
| 效果 | 好 | 更好（可组合使用）|
| 超参数 | patience | 正则化强度 |

**最佳实践**：Early Stopping + 轻度正则化，既省时间又防过拟合。

## 六、数据增强：让数据"无中生有"

### 6.1 核心思想

过拟合的本质是**数据太少**。如果能让数据"变多"，就能缓解过拟合。

**数据增强**：通过对训练数据进行**合理变换**，生成新的训练样本。

### 6.2 常见增强技术

**图像数据**：
- 旋转、翻转、裁剪
- 亮度、对比度调整
- 添加噪声

**文本数据**：
- 同义词替换
- 随机删除/交换词语
- 回译（翻译到另一语言再翻译回来）

**数值数据**：
- 添加高斯噪声
- 随机缩放
- SMOTE（合成少数类样本）

![数据增强技术](https://kimi-web-img.moonshot.cn/img/research.aimultiple.com/36427d55847e4354a6b613204535a390a8b43860.png)

### 6.3 关键原则

**增强必须是"合理"的**——变换后的样本仍然属于同一类别。

**错误示例**：
- 数字识别：把"6"旋转180度变成"9"
- 文本分类：把"不高兴"中的"不"删除变成"高兴"

## 七、交叉验证：更可靠的模型评估

### 7.1 为什么需要交叉验证？

**传统做法**：训练集70% / 测试集30%
**问题**：划分方式不同，结果差异很大

**K折交叉验证**：
1. 将数据分成K份（如5份）
2. 轮流用K-1份训练，1份验证
3. 取K次验证结果的平均

### 7.2 实现步骤

```
将数据集分成5份：D1, D2, D3, D4, D5

第1轮：用D2-D5训练，D1验证 → 准确率acc1
第2轮：用D1,D3-D5训练，D2验证 → 准确率acc2
...
第5轮：用D1-D4训练，D5验证 → 准确率acc5

最终准确率 = (acc1 + acc2 + acc3 + acc4 + acc5) / 5
```

**优点**：
- 所有数据都参与了训练和验证
- 结果更稳定，不受划分方式影响
- 能估计模型的方差

## 八、Python实现：正则化技术对比

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

# 生成带有噪声的数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + np.random.normal(0, 0.3, 100)  # 添加噪声

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用高阶多项式特征（容易过拟合）
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 训练不同模型
models = {
    'Linear (No Reg)': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1)
}

results = {}
for name, model in models.items():
    model.fit(X_train_poly, y_train)
    
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    results[name] = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse
    }
    
    print(f"{name}:")
    print(f"  训练MSE: {train_mse:.4f}")
    print(f"  测试MSE: {test_mse:.4f}")
    print(f"  差距: {test_mse - train_mse:.4f}")
    print()

# 可视化
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    
    # 绘制数据点
    ax.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
    ax.scatter(X_test, y_test, color='green', alpha=0.5, label='Test')
    
    # 绘制拟合曲线
    y_plot = result['model'].predict(X_plot_poly)
    ax.plot(X_plot, y_plot, color='red', linewidth=2, label='Fit')
    ax.plot(X_plot, np.sin(X_plot), color='black', linestyle='--', label='True')
    
    ax.set_title(f"{name}\nTrain: {result['train_mse']:.3f}, Test: {result['test_mse']:.3f}")
    ax.legend()
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show()

# 观察L1的稀疏性
lasso_coef = results['Lasso (L1)']['model'].coef_
print(f"Lasso非零系数数量: {np.sum(lasso_coef != 0)} / {len(lasso_coef)}")
print(f"Ridge非零系数数量: {np.sum(results['Ridge (L2)']['model'].coef_ != 0)} / {len(lasso_coef)}")
```

## 九、C/C++实现：带L2正则化的神经网络

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define N_SAMPLES 100
#define EPOCHS 1000
#define LEARNING_RATE 0.01
#define LAMBDA 0.001  // L2正则化强度

// 简单的二分类数据集（带噪声）
double X[N_SAMPLES][INPUT_SIZE];
double y[N_SAMPLES];

// 网络参数
double W1[HIDDEN_SIZE][INPUT_SIZE];
double b1[HIDDEN_SIZE];
double W2[OUTPUT_SIZE][HIDDEN_SIZE];
double b2[OUTPUT_SIZE];

// 初始化数据（XOR的变体，带噪声）
void init_data() {
    srand(42);
    for (int i = 0; i < N_SAMPLES; i++) {
        int x1 = rand() % 2;
        int x2 = rand() % 2;
        X[i][0] = x1 + (rand() / (double)RAND_MAX - 0.5) * 0.2;  // 加噪声
        X[i][1] = x2 + (rand() / (double)RAND_MAX - 0.5) * 0.2;
        y[i] = (x1 != x2) ? 1.0 : 0.0;  // XOR
    }
}

// Xavier初始化
void init_weights() {
    double scale1 = sqrt(2.0 / INPUT_SIZE);
    double scale2 = sqrt(2.0 / HIDDEN_SIZE);
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale1;
        }
        b1[i] = 0.0;
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale2;
        }
        b2[i] = 0.0;
    }
}

double sigmoid(double x) {
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

// 前向传播（单个样本）
void forward(double x[], double h[], double *output) {
    // 隐藏层
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            h[i] += W1[i][j] * x[j];
        }
        h[i] = sigmoid(h[i]);
    }
    
    // 输出层
    *output = b2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        *output += W2[0][i] * h[i];
    }
    *output = sigmoid(*output);
}

// 计算带L2正则化的损失
double compute_loss(int sample_idx) {
    double h[HIDDEN_SIZE], output;
    forward(X[sample_idx], h, &output);
    
    double bce_loss = -(y[sample_idx] * log(output + 1e-15) + 
                       (1 - y[sample_idx]) * log(1 - output + 1e-15));
    
    // L2正则化项
    double l2_loss = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            l2_loss += W1[i][j] * W1[i][j];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            l2_loss += W2[i][j] * W2[i][j];
        }
    }
    
    return bce_loss + 0.5 * LAMBDA * l2_loss;
}

// 训练（带L2正则化的梯度下降）
void train() {
    printf("训练带L2正则化的神经网络...\n");
    printf("正则化强度 lambda = %.4f\n\n", LAMBDA);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        
        for (int s = 0; s < N_SAMPLES; s++) {
            // 前向传播
            double h[HIDDEN_SIZE], output;
            forward(X[s], h, &output);
            
            total_loss += compute_loss(s);
            
            // 反向传播
            double doutput = output - y[s];
            
            // 输出层梯度（包含L2正则化梯度）
            double dW2[OUTPUT_SIZE][HIDDEN_SIZE];
            double db2[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                db2[i] = doutput;
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    dW2[i][j] = doutput * h[j] + LAMBDA * W2[i][j];  // L2梯度
                }
            }
            
            // 隐藏层梯度
            double dh[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                dh[i] = doutput * W2[0][i] * sigmoid_derivative(h[i]);
            }
            
            double dW1[HIDDEN_SIZE][INPUT_SIZE];
            double db1[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                db1[i] = dh[i];
                for (int j = 0; j < INPUT_SIZE; j++) {
                    dW1[i][j] = dh[i] * X[s][j] + LAMBDA * W1[i][j];  // L2梯度
                }
            }
            
            // 更新参数
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                b2[i] -= LEARNING_RATE * db2[i];
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    W2[i][j] -= LEARNING_RATE * dW2[i][j];
                }
            }
            
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                b1[i] -= LEARNING_RATE * db1[i];
                for (int j = 0; j < INPUT_SIZE; j++) {
                    W1[i][j] -= LEARNING_RATE * dW1[i][j];
                }
            }
        }
        
        if (epoch % 100 == 0) {
            printf("Epoch %d: Loss = %.4f\n", epoch, total_loss / N_SAMPLES);
        }
    }
}

// 计算准确率
double compute_accuracy() {
    int correct = 0;
    for (int s = 0; s < N_SAMPLES; s++) {
        double h[HIDDEN_SIZE], output;
        forward(X[s], h, &output);
        int pred = (output >= 0.5) ? 1 : 0;
        int true_label = (y[s] >= 0.5) ? 1 : 0;
        if (pred == true_label) correct++;
    }
    return (double)correct / N_SAMPLES;
}

int main() {
    init_data();
    init_weights();
    
    printf("数据集: 带噪声的XOR问题 (%d样本)\n", N_SAMPLES);
    printf("网络结构: %d-%d-%d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    train();
    
    double acc = compute_accuracy();
    printf("\n训练完成! 准确率: %.2f%%\n", acc * 100);
    
    // 打印权重范数（观察L2正则化效果）
    double w1_norm = 0.0, w2_norm = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            w1_norm += W1[i][j] * W1[i][j];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            w2_norm += W2[i][j] * W2[i][j];
        }
    }
    printf("W1 L2范数: %.4f\n", sqrt(w1_norm));
    printf("W2 L2范数: %.4f\n", sqrt(w2_norm));
    
    return 0;
}
```

## 十、Java实现：带Dropout的MLP

```java
import java.util.Random;

public class RegularizedMLP {
    
    private int inputSize, hiddenSize, outputSize;
    private double learningRate;
    private double dropoutRate;
    private Random random;
    
    // 参数
    private double[][] W1, W2;
    private double[] b1, b2;
    
    // Dropout掩码
    private boolean[] dropoutMask;
    
    public RegularizedMLP(int inputSize, int hiddenSize, int outputSize, 
                          double learningRate, double dropoutRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.dropoutRate = dropoutRate;
        this.random = new Random(42);
        
        initializeWeights();
    }
    
    private void initializeWeights() {
        double scale1 = Math.sqrt(2.0 / inputSize);
        double scale2 = Math.sqrt(2.0 / hiddenSize);
        
        W1 = new double[hiddenSize][inputSize];
        b1 = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] = (random.nextDouble() - 0.5) * 2 * scale1;
            }
        }
        
        W2 = new double[outputSize][hiddenSize];
        b2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] = (random.nextDouble() - 0.5) * 2 * scale2;
            }
        }
        
        dropoutMask = new boolean[hiddenSize];
    }
    
    private double sigmoid(double x) {
        if (x > 500) return 1.0;
        if (x < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    private double sigmoidDerivative(double a) {
        return a * (1.0 - a);
    }
    
    // 前向传播（训练时带Dropout，测试时不带）
    public double[] forward(double[] x, boolean isTraining) {
        // 隐藏层
        double[] h = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            h[i] = b1[i];
            for (int j = 0; j < inputSize; j++) {
                h[i] += W1[i][j] * x[j];
            }
            h[i] = sigmoid(h[i]);
            
            // Dropout（仅训练时）
            if (isTraining) {
                dropoutMask[i] = random.nextDouble() > dropoutRate;
                if (dropoutMask[i]) {
                    h[i] /= (1.0 - dropoutRate);  // 缩放
                } else {
                    h[i] = 0.0;  // 丢弃
                }
            }
        }
        
        // 输出层（不Dropout）
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += W2[i][j] * h[j];
            }
            output[i] = sigmoid(output[i]);
        }
        
        return output;
    }
    
    // 训练单个样本
    public void train(double[] x, double[] y) {
        // 前向传播（训练模式，启用Dropout）
        double[] h = new double[hiddenSize];
        
        // 手动前向以保存中间结果
        for (int i = 0; i < hiddenSize; i++) {
            h[i] = b1[i];
            for (int j = 0; j < inputSize; j++) {
                h[i] += W1[i][j] * x[j];
            }
            h[i] = sigmoid(h[i]);
            
            dropoutMask[i] = random.nextDouble() > dropoutRate;
            if (dropoutMask[i]) {
                h[i] /= (1.0 - dropoutRate);
            } else {
                h[i] = 0.0;
            }
        }
        
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += W2[i][j] * h[j];
            }
            output[i] = sigmoid(output[i]);
        }
        
        // 反向传播
        double[] dOutput = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            dOutput[i] = output[i] - y[i];
        }
        
        // 输出层梯度
        double[][] dW2 = new double[outputSize][hiddenSize];
        double[] db2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            db2[i] = dOutput[i];
            for (int j = 0; j < hiddenSize; j++) {
                dW2[i][j] = dOutput[i] * h[j];
            }
        }
        
        // 隐藏层梯度（考虑Dropout）
        double[] dH = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            if (!dropoutMask[i]) continue;  // 被丢弃的神经元不更新
            
            for (int j = 0; j < outputSize; j++) {
                dH[i] += dOutput[j] * W2[j][i];
            }
            dH[i] *= sigmoidDerivative(h[i] * (1.0 - dropoutRate));  // 反缩放
        }
        
        double[][] dW1 = new double[hiddenSize][inputSize];
        double[] db1 = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            db1[i] = dH[i];
            for (int j = 0; j < inputSize; j++) {
                dW1[i][j] = dH[i] * x[j];
            }
        }
        
        // 更新参数
        for (int i = 0; i < outputSize; i++) {
            b2[i] -= learningRate * db2[i];
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] -= learningRate * dW2[i][j];
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            b1[i] -= learningRate * db1[i];
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] -= learningRate * dW1[i][j];
            }
        }
    }
    
    // 预测（测试模式，无Dropout）
    public double[] predict(double[] x) {
        return forward(x, false);
    }
    
    public static void main(String[] args) {
        // XOR数据集
        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] y = {{0}, {1}, {1}, {0}};
        
        // 带Dropout的MLP
        RegularizedMLP mlp = new RegularizedMLP(2, 8, 1, 0.1, 0.5);
        
        System.out.println("训练带Dropout的MLP...");
        System.out.println("Dropout率: 0.5");
        
        // 训练多轮
        for (int epoch = 0; epoch < 5000; epoch++) {
            for (int i = 0; i < X.length; i++) {
                mlp.train(X[i], y[i]);
            }
        }
        
        // 测试（无Dropout）
        System.out.println("\n预测结果（测试模式，无Dropout）:");
        for (int i = 0; i < X.length; i++) {
            double[] pred = mlp.predict(X[i]);
            System.out.printf("输入: [%.0f, %.0f], 真实: %.0f, 预测: %.4f%n",
                X[i][0], X[i][1], y[i][0], pred[0]);
        }
    }
}
```

## 十一、五篇总结与深度学习准备

### 11.1 前五篇知识图谱

```
第1篇：线性回归
    ├── 损失函数（MSE）
    ├── 梯度下降
    └── 解析解 vs 迭代解

第2篇：逻辑回归
    ├── Sigmoid函数
    ├── 交叉熵损失
    └── 分类问题入门

第3篇：感知机与MLP
    ├── XOR问题
    ├── 非线性激活
    └── 反向传播

第4篇：优化算法
    ├── SGD → Momentum → Adam
    └── 学习率调度

第5篇：正则化（本文）
    ├── L1/L2正则化
    ├── Dropout
    ├── Early Stopping
    └── 数据增强与交叉验证
```

### 11.2 从ML到深度学习的桥梁

前五篇我们掌握了**机器学习的基础**：
- 模型：线性 → 非线性
- 优化：梯度下降及其改进
- 正则化：防止过拟合

**接下来（第6篇开始）**，我们将进入**真正的深度学习**：
- **CNN**：卷积神经网络，图像处理的王者
- **RNN**：循环神经网络，序列数据的专家
- **Transformer**：注意力机制，NLP的革命
- **GAN**：生成对抗网络，创造新数据

### 11.3 关键公式速查表

| 技术 | 公式/要点 | 使用场景 |
|:---|:---|:---|
| L2正则化 | $L + \lambda \sum w^2$ | 通用，权重收缩 |
| L1正则化 | $L + \lambda \sum \|w\|$ | 特征选择 |
| Dropout | 训练时以概率$p$丢弃神经元 | 深层网络 |
| Early Stopping | 验证损失不改善时停止 | 节省训练时间 |
| 数据增强 | 对训练数据进行合理变换 | 数据不足时 |

## 十二、实践建议：正则化使用指南

### 12.1 快速决策流程

```
模型过拟合了？
    │
    ├── 数据太少？ → 数据增强 / 收集更多数据
    │
    ├── 模型太复杂？ → Dropout / 减少网络层数
    │
    ├── 训练太久？ → Early Stopping
    │
    └── 权重太大？ → L2正则化（默认）/ L1正则化（需要特征选择）
```

### 12.2 超参数设置经验

| 技术 | 推荐值 | 调整方向 |
|:---|:---:|:---|
| L2正则化 $\lambda$ | 0.001 ~ 0.01 | 过拟合严重则增大 |
| L1正则化 $\lambda$ | 0.001 ~ 0.01 | 需要稀疏性则增大 |
| Dropout率 | 0.2 ~ 0.5 | 深层网络用高值 |
| Early Stopping patience | 10 ~ 20 | 训练不稳定则增大 |

### 12.3 组合使用策略

**轻度过拟合**：Early Stopping + 轻度L2（λ=0.001）
**中度过拟合**：Dropout(0.3) + L2(λ=0.01) + Early Stopping
**重度过拟合**：简化模型 + Dropout(0.5) + 数据增强 + L2(λ=0.1)

---

**本文代码已开源**：[深入探索深度学习的五大核心神经网络架(In depth exploration of the five core neural network architectures of deep learning)](https://github.com/city25/IDEOTFCNNARODL)

**配图清单：**
- 图1：过拟合与欠拟合对比图
- 图2：偏差-方差权衡曲线
- 图3：L1/L2正则化效果对比
- 图4：Dropout示意图
- 图5：Early Stopping原理图

---

*全文约5800字，是ML基础阶段的收官之作。前五篇我们打下了坚实基础，从第6篇开始，真正的深度学习之旅即将开启——CNN、RNN、Transformer、GAN，四大神经网络架构等你探索！*

---
> 本文部分内容由AI编辑，可能会出现幻觉，请谨慎阅读。
