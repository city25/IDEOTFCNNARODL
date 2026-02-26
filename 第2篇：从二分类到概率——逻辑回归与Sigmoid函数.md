# 第2篇：从二分类到概率——逻辑回归与Sigmoid函数

## 一、新问题：预测"是"或"否"

上一篇我们解决了**回归问题**——预测连续值（房价）。现在面对一个不同类型的挑战：

假设你是一家银行的风控专员，手头有以下客户数据：

| 月收入(万元) | 征信评分 | 是否有房贷 | **是否违约** |
|:---:|:---:|:---:|:---:|
| 1.5 | 720 | 否 | **否** |
| 0.8 | 580 | 是 | **是** |
| 2.2 | 780 | 否 | **否** |
| 1.0 | 620 | 是 | **是** |
| 1.8 | 750 | 否 | **否** |

现在一位新客户来了：月收入1.2万元，征信评分650，有房贷。问题是：**他会违约吗？**

这就是**二分类问题（Binary Classification）**。输出不再是连续数值，而是两个离散类别之一（是/否、0/1、正类/负类）。

**为什么不能直接用线性回归？**

假设我们强行用线性回归，让模型输出0或1。但线性回归的预测范围是 $(-\infty, +\infty)$，可能输出-0.5或1.3，这既不是有效概率，也难以解释。

我们需要一个函数，能够把任意实数"压缩"到0和1之间，并且具有概率的语义。

## 二、Sigmoid函数：从实数到概率的桥梁

### 2.1 函数定义

Sigmoid函数的数学表达式为：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

其中 $e$ 是自然常数（约2.71828），$z$ 是任意实数。

![Sigmoid函数图像](https://i-blog.csdnimg.cn/img_convert/90d3d5c2ff42455914379ad01e944aa6.jpeg)

### 2.2 函数特性

让我们一步步分析这个函数的行为：

**当 $z$ 很大时（比如 $z = 5$）：**
$$e^{-5} \approx 0.0067$$
$$\sigma(5) = \frac{1}{1 + 0.0067} \approx 0.993$$

**当 $z = 0$ 时：**
$$e^{0} = 1$$
$$\sigma(0) = \frac{1}{1 + 1} = 0.5$$

**当 $z$ 很小时（比如 $z = -5$）：**
$$e^{5} \approx 148.4$$
$$\sigma(-5) = \frac{1}{1 + 148.4} \approx 0.0067$$

**关键洞察：**
- 输出范围始终在 $(0, 1)$ 之间
- 当 $z = 0$ 时，输出恰好是0.5（分界点）
- 当 $z \to +\infty$，输出趋近于1
- 当 $z \to -\infty$，输出趋近于0
- 函数关于点 $(0, 0.5)$ 中心对称

### 2.3 导数的优雅性质

Sigmoid函数的导数有一个非常优美的形式：

$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

**推导过程：**

从 $\sigma(z) = (1 + e^{-z})^{-1}$ 开始，使用链式法则：

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z})$$
$$= \frac{e^{-z}}{(1 + e^{-z})^2}$$

现在，我们把分子改写成：
$$e^{-z} = (1 + e^{-z}) - 1$$

所以：
$$\sigma'(z) = \frac{(1 + e^{-z}) - 1}{(1 + e^{-z})^2}$$
$$= \frac{1}{1 + e^{-z}} - \frac{1}{(1 + e^{-z})^2}$$
$$= \sigma(z) - \sigma(z)^2$$
$$= \sigma(z)(1 - \sigma(z))$$

**这个性质在反向传播中极其重要！** 它让我们可以用函数值本身来计算导数，无需重复计算指数。

## 三、逻辑回归模型架构

### 3.1 模型定义

逻辑回归的完整表达式：

$$z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = \mathbf{w}^T\mathbf{x} + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

其中：
- $\mathbf{x}$ 是特征向量（输入）
- $\mathbf{w}$ 是权重向量
- $b$ 是偏置
- $\hat{y}$ 是预测为正类的概率

**解读**：模型先计算特征的线性组合 $z$，然后通过Sigmoid函数将其转换为0到1之间的概率值。

### 3.2 决策边界

我们如何根据概率做出分类决策？

通常设定阈值为0.5：
- 如果 $\hat{y} \geq 0.5$，预测为正类（1）
- 如果 $\hat{y} < 0.5$，预测为负类（0）

由于Sigmoid函数在 $z=0$ 时输出0.5，决策边界就是：

$$w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = 0$$

这是一个**线性决策边界**，将特征空间分成两个区域。

![决策边界示意图](https://i-blog.csdnimg.cn/img_convert/c856af56c3b2413c06674efc285a2b00.png)

## 四、损失函数：为什么不用均方误差？

### 4.1 均方误差的问题

如果我们像线性回归一样使用MSE：

$$L = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2$$

这个损失函数对于逻辑回归是**非凸的**，意味着它可能有多个局部最小值，梯度下降容易陷入局部最优。

### 4.2 最大似然估计的视角

换个角度思考：我们希望模型输出的概率尽可能接近真实标签。

对于单个样本：
- 如果真实标签 $y = 1$，我们希望 $\hat{y}$ 尽可能大（接近1）
- 如果真实标签 $y = 0$，我们希望 $\hat{y}$ 尽可能小（接近0）

可以把这统一写成：
$$P(y|x) = \hat{y}^y \cdot (1-\hat{y})^{1-y}$$

验证一下：
- 当 $y=1$：$P = \hat{y}^1 \cdot (1-\hat{y})^0 = \hat{y}$ ✓
- 当 $y=0$：$P = \hat{y}^0 \cdot (1-\hat{y})^1 = 1-\hat{y}$ ✓

对于整个数据集，假设样本独立，联合概率是各样本概率的乘积：

$$P(\text{data}) = \prod_{i=1}^{n} \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i}$$

**最大似然估计**的目标是最大化这个概率。为了方便计算，我们取对数（对数不改变极值点位置）：

$$\log P = \sum_{i=1}^{n} [y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)]$$

最大化似然等价于最小化负对数似然，于是得到**交叉熵损失（Cross-Entropy Loss）**：

$$L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)]$$

### 4.3 交叉熵的直观理解

考虑一个样本：

**情况1：$y=1$，模型预测 $\hat{y}=0.9$（正确且自信）**
$$L = -\log(0.9) \approx 0.105$$

**情况2：$y=1$，模型预测 $\hat{y}=0.1$（错误且自信）**
$$L = -\log(0.1) \approx 2.303$$

**情况3：$y=1$，模型预测 $\hat{y}=0.5$（不确定）**
$$L = -\log(0.5) \approx 0.693$$

**关键洞察**：当模型预测错误且很"自信"时，损失会非常大（因为对数函数在接近0时趋向负无穷）。这给了模型强烈的"纠错信号"。

## 五、梯度推导：反向传播的核心

我们需要计算损失函数对参数 $w_j$ 和 $b$ 的偏导数。

### 5.1 链式法则的应用

损失函数：$L = -\frac{1}{n} \sum [y \log \hat{y} + (1-y) \log (1-\hat{y})]$

其中 $\hat{y} = \sigma(z)$，$z = \mathbf{w}^T\mathbf{x} + b$

根据链式法则：
$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$

### 5.2 一步步计算

**第一步：$\frac{\partial L}{\partial \hat{y}}$**

$$L = -[y \log \hat{y} + (1-y) \log (1-\hat{y})]$$

$$\frac{\partial L}{\partial \hat{y}} = -[\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}] = -\frac{y(1-\hat{y}) - (1-y)\hat{y}}{\hat{y}(1-\hat{y})}$$
$$= -\frac{y - y\hat{y} - \hat{y} + y\hat{y}}{\hat{y}(1-\hat{y})} = -\frac{y - \hat{y}}{\hat{y}(1-\hat{y})} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

**第二步：$\frac{\partial \hat{y}}{\partial z}$**

这正是Sigmoid的导数：
$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$$

**第三步：$\frac{\partial z}{\partial w_j}$**

$$z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$$

$$\frac{\partial z}{\partial w_j} = x_j$$

### 5.3 合并结果

$$\frac{\partial L}{\partial w_j} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x_j = (\hat{y} - y) \cdot x_j$$

**惊人的简化！** 复杂的分式全部消去，最终形式异常简洁。

同理：
$$\frac{\partial L}{\partial b} = \hat{y} - y$$

### 5.4 向量化形式

对于所有样本：
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \mathbf{x}_i$$

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

注意这里的 $\frac{1}{n}$ 是因为损失函数定义中的平均。

## 六、Python实现：完整的训练流程

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid函数，使用clip防止溢出"""
        z = np.clip(z, -500, 500)  # 防止exp溢出
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """计算交叉熵损失"""
        # 防止log(0)错误
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 打印进度
            if epoch % 100 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    
    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def accuracy(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# 准备数据：银行贷款违约预测
# 特征：[月收入(万元), 征信评分/1000]
X = np.array([
    [1.5, 0.72],
    [0.8, 0.58],
    [2.2, 0.78],
    [1.0, 0.62],
    [1.8, 0.75],
    [0.9, 0.55],
    [2.0, 0.80],
    [1.1, 0.60],
    [1.6, 0.74],
    [0.7, 0.50]
])

# 标签：0=不违约, 1=违约
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

print("开始训练逻辑回归模型...")
model = LogisticRegression(learning_rate=0.5, epochs=1000)
model.fit(X, y)

# 测试新样本
X_test = np.array([[1.2, 0.65]])  # 月收入1.2万，征信650
prob = model.predict_proba(X_test)[0]
pred = model.predict(X_test)[0]

print(f"\n测试样本: 月收入1.2万元, 征信评分650")
print(f"违约概率: {prob:.4f}")
print(f"预测结果: {'违约' if pred == 1 else '不违约'}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：数据点和决策边界
ax1 = axes[0]
colors = ['blue' if label == 0 else 'red' for label in y]
ax1.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black')

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax1.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')
ax1.set_xlabel('月收入(万元)')
ax1.set_ylabel('征信评分(归一化)')
ax1.set_title('决策边界与概率分布')

# 图2：损失下降曲线
ax2 = axes[1]
ax2.plot(model.loss_history)
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('交叉熵损失')
ax2.set_title('训练过程')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 七、C/C++实现：深入数值计算

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_SAMPLES 10
#define N_FEATURES 2
#define EPOCHS 1000
#define LEARNING_RATE 0.5

// 数据集：银行贷款违约预测
// 特征1: 月收入(万元), 特征2: 征信评分/1000
double X[N_SAMPLES][N_FEATURES] = {
    {1.5, 0.72}, {0.8, 0.58}, {2.2, 0.78}, {1.0, 0.62}, {1.8, 0.75},
    {0.9, 0.55}, {2.0, 0.80}, {1.1, 0.60}, {1.6, 0.74}, {0.7, 0.50}
};

// 标签: 0=不违约, 1=违约
double y[N_SAMPLES] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

// 模型参数
double weights[N_FEATURES] = {0.0, 0.0};
double bias = 0.0;

// Sigmoid函数
double sigmoid(double z) {
    // 防止exp溢出
    if (z > 500) return 1.0;
    if (z < -500) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

// 计算预测值
double predict_one(double x[], double w[], double b) {
    double z = 0.0;
    for (int j = 0; j < N_FEATURES; j++) {
        z += w[j] * x[j];
    }
    z += b;
    return sigmoid(z);
}

// 计算交叉熵损失
double compute_loss() {
    double loss = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double y_pred = predict_one(X[i], weights, bias);
        // 防止log(0)
        if (y_pred < 1e-15) y_pred = 1e-15;
        if (y_pred > 1 - 1e-15) y_pred = 1 - 1e-15;
        
        loss -= y[i] * log(y_pred) + (1 - y[i]) * log(1 - y_pred);
    }
    return loss / N_SAMPLES;
}

// 计算准确率
double compute_accuracy() {
    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double prob = predict_one(X[i], weights, bias);
        int pred = (prob >= 0.5) ? 1 : 0;
        if (pred == (int)y[i]) correct++;
    }
    return (double)correct / N_SAMPLES;
}

// 训练模型
void train() {
    printf("开始训练逻辑回归模型...\n");
    printf("%-10s %-15s %-15s\n", "Epoch", "Loss", "Accuracy");
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // 存储所有预测值
        double y_pred[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) {
            y_pred[i] = predict_one(X[i], weights, bias);
        }
        
        // 计算梯度
        double dw[N_FEATURES] = {0.0};
        double db = 0.0;
        
        for (int i = 0; i < N_SAMPLES; i++) {
            double error = y_pred[i] - y[i];
            for (int j = 0; j < N_FEATURES; j++) {
                dw[j] += error * X[i][j];
            }
            db += error;
        }
        
        // 平均梯度
        for (int j = 0; j < N_FEATURES; j++) {
            dw[j] /= N_SAMPLES;
        }
        db /= N_SAMPLES;
        
        // 更新参数
        for (int j = 0; j < N_FEATURES; j++) {
            weights[j] -= LEARNING_RATE * dw[j];
        }
        bias -= LEARNING_RATE * db;
        
        // 打印进度
        if (epoch % 100 == 0) {
            double loss = compute_loss();
            double acc = compute_accuracy();
            printf("%-10d %-15.4f %-15.4f\n", epoch, loss, acc);
        }
    }
}

int main() {
    train();
    
    printf("\n训练完成!\n");
    printf("权重: w1=%.4f, w2=%.4f\n", weights[0], weights[1]);
    printf("偏置: b=%.4f\n", bias);
    
    // 测试新样本
    double x_test[N_FEATURES] = {1.2, 0.65};  // 月收入1.2万，征信650
    double prob = predict_one(x_test, weights, bias);
    int pred = (prob >= 0.5) ? 1 : 0;
    
    printf("\n测试样本: 月收入1.2万元, 征信评分650\n");
    printf("违约概率: %.4f\n", prob);
    printf("预测结果: %s\n", pred == 1 ? "违约" : "不违约");
    
    return 0;
}
```

**编译运行：**
```bash
gcc logistic_regression.c -o logistic_regression -lm
./logistic_regression
```

## 八、Java实现：面向对象的工程实践

```java
public class LogisticRegression {
    
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    
    // 训练数据
    private double[][] X;
    private double[] y;
    private int nSamples;
    private int nFeatures;
    
    public LogisticRegression(double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
    }
    
    // Sigmoid函数
    private double sigmoid(double z) {
        if (z > 500) return 1.0;
        if (z < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    // 单个样本预测
    private double predictOne(double[] x) {
        double z = bias;
        for (int j = 0; j < nFeatures; j++) {
            z += weights[j] * x[j];
        }
        return sigmoid(z);
    }
    
    // 计算交叉熵损失
    private double computeLoss() {
        double loss = 0.0;
        for (int i = 0; i < nSamples; i++) {
            double yPred = predictOne(X[i]);
            // 数值稳定性处理
            yPred = Math.max(1e-15, Math.min(1 - 1e-15, yPred));
            loss -= y[i] * Math.log(yPred) + (1 - y[i]) * Math.log(1 - yPred);
        }
        return loss / nSamples;
    }
    
    // 计算准确率
    private double computeAccuracy() {
        int correct = 0;
        for (int i = 0; i < nSamples; i++) {
            double prob = predictOne(X[i]);
            int pred = (prob >= 0.5) ? 1 : 0;
            if (pred == (int)y[i]) correct++;
        }
        return (double) correct / nSamples;
    }
    
    // 训练模型
    public void fit(double[][] X, double[] y) {
        this.X = X;
        this.y = y;
        this.nSamples = X.length;
        this.nFeatures = X[0].length;
        
        // 初始化参数
        this.weights = new double[nFeatures];
        this.bias = 0.0;
        
        System.out.println("开始训练逻辑回归模型...");
        System.out.printf("%-10s %-15s %-15s%n", "Epoch", "Loss", "Accuracy");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // 计算所有预测值
            double[] yPred = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                yPred[i] = predictOne(X[i]);
            }
            
            // 计算梯度
            double[] dw = new double[nFeatures];
            double db = 0.0;
            
            for (int i = 0; i < nSamples; i++) {
                double error = yPred[i] - y[i];
                for (int j = 0; j < nFeatures; j++) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }
            
            // 平均梯度
            for (int j = 0; j < nFeatures; j++) {
                dw[j] /= nSamples;
            }
            db /= nSamples;
            
            // 更新参数
            for (int j = 0; j < nFeatures; j++) {
                weights[j] -= learningRate * dw[j];
            }
            bias -= learningRate * db;
            
            // 打印进度
            if (epoch % 100 == 0) {
                double loss = computeLoss();
                double acc = computeAccuracy();
                System.out.printf("%-10d %-15.4f %-15.4f%n", epoch, loss, acc);
            }
        }
    }
    
    // 预测概率
    public double predictProba(double[] x) {
        return predictOne(x);
    }
    
    // 预测类别
    public int predict(double[] x) {
        return (predictOne(x) >= 0.5) ? 1 : 0;
    }
    
    // 获取参数
    public double[] getWeights() {
        return weights;
    }
    
    public double getBias() {
        return bias;
    }
    
    // 主函数
    public static void main(String[] args) {
        // 训练数据
        double[][] X = {
            {1.5, 0.72}, {0.8, 0.58}, {2.2, 0.78}, {1.0, 0.62}, {1.8, 0.75},
            {0.9, 0.55}, {2.0, 0.80}, {1.1, 0.60}, {1.6, 0.74}, {0.7, 0.50}
        };
        
        double[] y = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
        
        LogisticRegression model = new LogisticRegression(0.5, 1000);
        model.fit(X, y);
        
        System.out.println("\n训练完成!");
        double[] w = model.getWeights();
        System.out.printf("权重: w1=%.4f, w2=%.4f%n", w[0], w[1]);
        System.out.printf("偏置: b=%.4f%n", model.getBias());
        
        // 测试
        double[] xTest = {1.2, 0.65};
        double prob = model.predictProba(xTest);
        int pred = model.predict(xTest);
        
        System.out.printf("%n测试样本: 月收入1.2万元, 征信评分650%n");
        System.out.printf("违约概率: %.4f%n", prob);
        System.out.printf("预测结果: %s%n", pred == 1 ? "违约" : "不违约");
    }
}
```

## 九、多分类扩展：Softmax与One-vs-Rest

逻辑回归本质上是二分类器。如何处理多分类问题（比如预测客户信用等级：A/B/C/D）？

### 9.1 One-vs-Rest (OvR) 策略

训练多个二分类器：
- 分类器1：A类 vs 非A类
- 分类器2：B类 vs 非B类
- 分类器3：C类 vs 非C类
- 分类器4：D类 vs 非D类

预测时，选择概率最高的那个分类器对应的类别。

### 9.2 Softmax回归（多分类逻辑回归）

更优雅的方法是Softmax回归，直接输出每个类别的概率：

$$P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

其中 $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$，$K$ 是类别数。

Softmax确保所有类别概率之和为1，是神经网络多分类输出的标准选择。

## 十、总结与核心洞察

### 10.1 关键公式回顾

| 概念 | 公式 |
|:---:|:---|
| Sigmoid函数 | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| 模型输出 | $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$ |
| 交叉熵损失 | $L = -\frac{1}{n}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| 梯度 | $\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum(\hat{y}_i - y_i)x_{ij}$ |

### 10.2 与线性回归的对比

| 特性 | 线性回归 | 逻辑回归 |
|:---:|:---:|:---:|
| 问题类型 | 回归（连续值） | 分类（离散值） |
| 输出范围 | $(-\infty, +\infty)$ | $(0, 1)$ |
| 激活函数 | 无（线性） | Sigmoid |
| 损失函数 | 均方误差(MSE) | 交叉熵(Cross-Entropy) |
| 解释 | 预测值 | 概率 |

### 10.3 神经网络的前奏

逻辑 regression是**最简单的神经网络**：
- 输入层：特征 $\mathbf{x}$
- 权重连接：$\mathbf{w}$
- 激活函数：Sigmoid
- 输出层：概率 $\hat{y}$

![逻辑回归作为神经网络](https://i-blog.csdnimg.cn/img_convert/f5796588624400f75bb8b08a2b11d9fc.png)

当我们把多个这样的单元堆叠起来，就形成了**多层感知机（MLP）**，也就是我们下一篇要讲的**神经网络**。

## 十一、练习题与思考

1. **数学推导**：证明Sigmoid函数的导数 $\sigma'(z) = \sigma(z)(1-\sigma(z))$

2. **代码实现**：修改Python代码，实现L2正则化（在损失函数中加入 $\lambda \sum w_j^2$）

3. **可视化**：绘制Sigmoid函数在不同参数（缩放和平移）下的图像，理解 $w$ 和 $b$ 如何影响决策边界

4. **思考题**：为什么逻辑回归使用交叉熵损失而不是均方误差？从梯度的角度分析（提示：考虑当预测错误且置信度高时，两种损失的梯度大小）

**下一篇预告：《第3篇：从单层到多层——感知机与多层感知机的突破》**

我们将解决逻辑回归的局限——只能处理线性可分问题。通过堆叠多个神经元，引入非线性激活函数，我们将构建第一个真正的**神经网络**，并理解为什么"深度"如此重要。

---

**本文代码已开源**：[深入探索深度学习的五大核心神经网络架(In depth exploration of the five core neural network architectures of deep learning)](https://github.com/city25/IDEOTFCNNARODL)

**配图清单：**
- 图1：Sigmoid函数曲线及其特性
- 图2：逻辑回归决策边界示意图
- 图3：线性回归vs逻辑回归对比图
- 图4：Python训练过程可视化
- 图5：逻辑回归作为单层神经网络结构图

---

*全文约5200字，涵盖理论推导、算法实现、代码详解和扩展思考。建议读者完成课后练习，特别是手动推导梯度公式，这对理解反向传播至关重要。*

---
> 本文部分内容由AI编辑，可能会出现幻觉，请谨慎阅读。