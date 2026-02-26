# 第4篇：优化之道——从梯度下降到Adam的进化之路

## 一、一个困扰所有初学者的问题

还记得我们在第3篇中实现的XOR神经网络吗？

当你运行那段代码时，可能遇到过这样的困惑：
- 为什么有时候损失降到一半就不动了？
- 为什么学习率设为0.5能收敛，设为0.1却慢得像蜗牛？
- 为什么同样的网络，有时10轮就收敛，有时1000轮还在震荡？

2012年，Alex Krizhevsky在训练AlexNet时，也遇到了同样的问题。他的解决方案——**Adam优化器**——让深度学习真正进入了大规模应用时代。

**优化算法，是神经网络从"能跑"到"跑得快、跑得稳"的关键。**

本文将带你走过70年的优化算法进化史，从最简单的梯度下降，到现代深度学习的标配Adam，理解每一种算法背后的直觉和数学原理。

![优化算法对比](https://kimi-web-img.moonshot.cn/img/cdn.analyticsvidhya.com/77770af92ddd395ef651870e8a9e12d5b44900f7.webp)

## 二、标准梯度下降的问题

### 2.1 三种梯度下降策略

首先，我们需要明确梯度下降的三种形式：

**批量梯度下降（Batch GD）**：
$$\mathbf{w} = \mathbf{w} - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} \nabla L_i(\mathbf{w})$$

使用**全部数据**计算梯度，更新一次。稳定但慢。

**随机梯度下降（Stochastic GD）**：
$$\mathbf{w} = \mathbf{w} - \eta \cdot \nabla L_i(\mathbf{w})$$

每次用**一个样本**计算梯度，更新频繁。快但噪声大。

**小批量梯度下降（Mini-batch GD）**：
$$\mathbf{w} = \mathbf{w} - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla L_i(\mathbf{w})$$

使用**小批量（如32、64、128个样本）**。平衡了稳定性和速度，是实际最常用的方法。

### 2.2 标准梯度下降的四大痛点

**痛点1：学习率选择困难**
- 太大 → 震荡发散
- 太小 → 收敛极慢
- 刚刚好 → 很难找，且不同参数需要不同学习率

**痛点2：鞍点困境**
在深度网络中，损失函数存在大量**鞍点**（梯度为0但不是最小值）：
$$\frac{\partial L}{\partial w} = 0, \text{但 Hessian矩阵有正有负特征值}$$

标准梯度下降在鞍点会"卡住"，因为梯度为0就不再更新。

**痛点3：局部最优与病态曲率**
损失函数的某些方向很陡峭，某些方向很平缓，形成**峡谷地形**。梯度下降会在峡谷两侧来回震荡，沿着谷底前进缓慢。

**痛点4：不同参数需要不同更新幅度**
在神经网络中，不同层的参数梯度大小差异巨大：
- 靠近输入层的梯度通常较小（梯度消失）
- 靠近输出层的梯度通常较大

用**同一个学习率**更新所有参数显然不合理。

## 三、Momentum：给梯度下降加上惯性

### 3.1 物理直觉

想象一个球从山上滚下来：
- 在陡峭处，球加速下落
- 在平缓处，球依靠惯性继续前进
- 遇到小坑，球的惯性帮助它越过

**Momentum算法**就是给梯度下降加上这样的"惯性"。

### 3.2 算法推导

引入**速度变量** $\mathbf{v}$，记录历史梯度的累积：

$$\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \eta \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_t$$

其中：
- $\gamma$ 是动量系数（通常0.9），表示保留多少历史速度
- $\mathbf{v}$ 是速度（动量），是梯度的指数加权移动平均

**直观理解：**
- 当前梯度 $\nabla L$ 提供"加速度"
- 历史速度 $\mathbf{v}_{t-1}$ 提供"惯性"
- 两者结合，既响应当前梯度，又保持运动方向

### 3.3 为什么Momentum有效？

**1. 加速收敛**
在梯度方向一致的方向上（如沿着峡谷），速度不断累积，加速前进。

**2. 抑制震荡**
在梯度方向变化剧烈的方向上（如峡谷两侧），历史速度与新梯度部分抵消，减少震荡。

**3. 逃离局部最优和鞍点**
即使当前梯度为0（鞍点），历史速度 $\mathbf{v}$ 仍推动参数继续前进，有机会逃离。

## 四、AdaGrad：自适应学习率

### 4.1 核心思想

不同参数应该有不同的学习率：
- 梯度大的参数 → 学习率小一些（谨慎更新）
- 梯度小的参数 → 学习率大一些（加速更新）

### 4.2 算法推导

维护一个**梯度平方的累积和** $\mathbf{r}$：

$$\mathbf{r}_t = \mathbf{r}_{t-1} + (\nabla L(\mathbf{w}_t))^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{r}_t} + \epsilon} \odot \nabla L(\mathbf{w}_t)$$

其中：
- $\mathbf{r}$ 记录每个参数的历史梯度平方和
- 分母 $\sqrt{\mathbf{r}_t}$ 实现了自适应：历史梯度大 → 学习率衰减快
- $\epsilon$（通常1e-8）防止除零

**关键洞察**：频繁更新的参数（梯度大）学习率被压低，罕见更新的参数（梯度小）学习率保持较高。

### 4.3 AdaGrad的问题

**学习率单调递减**：由于 $\mathbf{r}$ 只增不减，学习率会不断衰减，最终可能完全停止学习。

这在深度网络中是个大问题——训练后期参数几乎不再更新。

## 五、RMSProp：改进的自适应学习率

### 5.1 解决AdaGrad的缺陷

RMSProp用**指数移动平均**代替累积和，让历史梯度"遗忘"旧信息：

$$\mathbf{r}_t = \rho \mathbf{r}_{t-1} + (1-\rho)(\nabla L(\mathbf{w}_t))^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{r}_t} + \epsilon} \odot \nabla L(\mathbf{w}_t)$$

其中 $\rho$ 通常设为0.9。

**对比AdaGrad：**
- AdaGrad：$\mathbf{r}_t = \sum_{i=1}^{t} g_i^2$（累积所有历史）
- RMSProp：$\mathbf{r}_t = 0.9\mathbf{r}_{t-1} + 0.1g_t^2$（只关注近期）

这样学习率不会无限衰减，可以持续学习。

## 六、Adam：Momentum + RMSProp的完美结合

### 6.1 为什么Adam成为默认选择？

**Adam（Adaptive Moment Estimation）**结合了：
- **Momentum**的一阶矩估计（均值）→ 提供惯性
- **RMSProp**的二阶矩估计（未中心化的方差）→ 自适应学习率

### 6.2 完整算法推导

**第一步：计算梯度**
$$\mathbf{g}_t = \nabla L(\mathbf{w}_t)$$

**第二步：更新一阶矩（动量）**
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$$

$\mathbf{m}_t$ 是梯度的指数移动平均，类似于Momentum中的速度。

**第三步：更新二阶矩（自适应）**
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$

$\mathbf{v}_t$ 是梯度平方的指数移动平均，类似于RMSProp中的 $\mathbf{r}$。

**第四步：偏差修正（关键！）**

由于 $\mathbf{m}$ 和 $\mathbf{v}$ 初始化为0，前几步的估计会偏向0。Adam通过偏差修正解决这个问题：

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}$$
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

**为什么需要修正？**

假设 $\beta_1 = 0.9$，初始 $\mathbf{m}_0 = 0$：
- 第1步：$\mathbf{m}_1 = 0.9 \cdot 0 + 0.1 \cdot g_1 = 0.1g_1$（只有真实值的10%）
- 修正后：$\hat{\mathbf{m}}_1 = \frac{0.1g_1}{1-0.9} = g_1$（恢复到真实值）

**第五步：参数更新**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

**默认超参数**：$\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

### 6.3 Adam的优势总结

| 特性 | 效果 |
|:---|:---|
| 动量项 $\mathbf{m}$ | 加速收敛，减少震荡 |
| 自适应项 $\mathbf{v}$ | 不同参数不同学习率 |
| 偏差修正 | 早期训练更稳定 |
| 计算高效 | 只需一阶导数，内存需求小 |

![Adam与其他优化器对比](https://kimi-web-img.moonshot.cn/img/machinelearningmastery.com/21708a4df79a91dfc7a858e02fa82e3a60ec6db5.png)

## 七、学习率调度：让学习率随时间变化

即使有了Adam，学习率的选择仍然重要。**学习率调度策略**让学习率随训练进程动态调整。

### 7.1 常用调度策略

**1. 阶梯衰减（Step Decay）**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{epoch\_drop} \rfloor}$$

每过一定轮数，学习率乘以衰减系数（如0.1）。

**2. 指数衰减（Exponential Decay）**
$$\eta_t = \eta_0 \cdot e^{-kt}$$

学习率连续指数下降。

**3. 余弦退火（Cosine Annealing）**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

学习率按余弦曲线变化，从大到小再到大（周期性重启）。

![学习率调度对比](https://kimi-web-img.moonshot.cn/img/i0.wp.com/092f064903f4a9cea0e8a9fbc50ca802b8dea542.png)

### 7.2 为什么学习率调度有效？

**训练初期**：大学习率 → 快速接近最优区域

**训练后期**：小学习率 → 精细调整，避免在最优点附近震荡

**周期性重启**：帮助跳出局部最优，探索更好的解

## 八、Python实现：优化器对比实验

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义测试函数：Rosenbrock函数（经典的优化测试函数，有狭长峡谷）
def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock函数，全局最小值在(1,1)"""
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    """Rosenbrock函数的梯度"""
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])

# 优化器基类
class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.history = []
    
    def step(self, params, grad):
        raise NotImplementedError
    
    def record(self, params, loss):
        self.history.append({'params': params.copy(), 'loss': loss})

# SGD
class SGD(Optimizer):
    def step(self, params, grad):
        return params - self.lr * grad

# SGD + Momentum
class Momentum(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
        self.v = None
    
    def step(self, params, grad):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.gamma * self.v + self.lr * grad
        return params - self.v

# AdaGrad
class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.r = None
    
    def step(self, params, grad):
        if self.r is None:
            self.r = np.zeros_like(params)
        self.r += grad ** 2
        return params - (self.lr / (np.sqrt(self.r) + self.epsilon)) * grad

# RMSProp
class RMSProp(Optimizer):
    def __init__(self, lr=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.r = None
    
    def step(self, params, grad):
        if self.r is None:
            self.r = np.zeros_like(params)
        self.r = self.rho * self.r + (1 - self.rho) * (grad ** 2)
        return params - (self.lr / (np.sqrt(self.r) + self.epsilon)) * grad

# Adam
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # 更新一阶矩和二阶矩
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 更新参数
        return params - (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat

# 训练函数
def train_optimizer(optimizer, init_params, n_steps=1000):
    params = init_params.copy()
    optimizer.history = []
    
    for step in range(n_steps):
        loss = rosenbrock(params[0], params[1])
        grad = rosenbrock_grad(params[0], params[1])
        optimizer.record(params, loss)
        params = optimizer.step(params, grad)
    
    return optimizer

# 实验对比
np.random.seed(42)
init_params = np.array([-1.0, 2.0])  # 初始点远离最优解(1,1)

optimizers = {
    'SGD': SGD(lr=0.001),
    'Momentum': Momentum(lr=0.001, gamma=0.9),
    'AdaGrad': AdaGrad(lr=0.1),
    'RMSProp': RMSProp(lr=0.01),
    'Adam': Adam(lr=0.01)
}

results = {}
for name, opt in optimizers.items():
    print(f"训练 {name}...")
    results[name] = train_optimizer(opt, init_params, n_steps=2000)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 绘制损失曲线
ax_loss = axes[0, 0]
for name, result in results.items():
    losses = [h['loss'] for h in result.history]
    ax_loss.plot(losses, label=name, linewidth=2)
ax_loss.set_xlabel('Steps')
ax_loss.set_ylabel('Loss')
ax_loss.set_yscale('log')
ax_loss.set_title('Loss Curves Comparison')
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)

# 绘制优化轨迹
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

for idx, (name, result) in enumerate(results.items()):
    ax = axes[(idx + 1) // 3, (idx + 1) % 3]
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.5, cmap='viridis')
    
    # 绘制轨迹
    trajectory = np.array([h['params'] for h in result.history])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1, alpha=0.7)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')
    ax.plot(1, 1, 'k+', markersize=15, label='Optimal')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{name} Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印最终结果
print("\n最终损失值对比:")
print("-" * 40)
for name, result in results.items():
    final_loss = result.history[-1]['loss']
    final_pos = result.history[-1]['params']
    print(f"{name:<12} Loss: {final_loss:.6f}  Position: ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
```

## 九、C/C++实现：Adam优化器

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_PARAMS 2
#define N_STEPS 2000
#define LR 0.01

// Rosenbrock函数及其梯度
double rosenbrock(double x, double y) {
    double a = 1.0, b = 100.0;
    return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
}

void rosenbrock_grad(double x, double y, double *dx, double *dy) {
    double a = 1.0, b = 100.0;
    *dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
    *dy = 2.0 * b * (y - x * x);
}

// 优化器结构
typedef struct {
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double m[N_PARAMS];
    double v[N_PARAMS];
    int t;
} AdamOptimizer;

// 初始化Adam
void adam_init(AdamOptimizer *opt, double lr) {
    opt->lr = lr;
    opt->beta1 = 0.9;
    opt->beta2 = 0.999;
    opt->epsilon = 1e-8;
    opt->t = 0;
    for (int i = 0; i < N_PARAMS; i++) {
        opt->m[i] = 0.0;
        opt->v[i] = 0.0;
    }
}

// Adam更新步骤
void adam_step(AdamOptimizer *opt, double *params, double *grad) {
    opt->t++;
    
    for (int i = 0; i < N_PARAMS; i++) {
        // 更新一阶矩
        opt->m[i] = opt->beta1 * opt->m[i] + (1 - opt->beta1) * grad[i];
        // 更新二阶矩
        opt->v[i] = opt->beta2 * opt->v[i] + (1 - opt->beta2) * grad[i] * grad[i];
        
        // 偏差修正
        double m_hat = opt->m[i] / (1.0 - pow(opt->beta1, opt->t));
        double v_hat = opt->v[i] / (1.0 - pow(opt->beta2, opt->t));
        
        // 更新参数
        params[i] -= (opt->lr / (sqrt(v_hat) + opt->epsilon)) * m_hat;
    }
}

int main() {
    // 初始参数
    double params[N_PARAMS] = {-1.0, 2.0};
    double grad[N_PARAMS];
    
    AdamOptimizer opt;
    adam_init(&opt, LR);
    
    printf("Adam优化器训练Rosenbrock函数\n");
    printf("初始位置: (%.4f, %.4f), 损失: %.6f\n", 
           params[0], params[1], rosenbrock(params[0], params[1]));
    printf("目标位置: (1.0000, 1.0000), 损失: 0.000000\n\n");
    
    printf("%-10s %-15s %-15s %-15s\n", "Step", "x", "y", "Loss");
    printf("-" * 55);
    
    for (int step = 0; step <= N_STEPS; step++) {
        double loss = rosenbrock(params[0], params[1]);
        
        if (step % 200 == 0) {
            printf("%-10d %-15.6f %-15.6f %-15.6f\n", 
                   step, params[0], params[1], loss);
        }
        
        // 计算梯度
        rosenbrock_grad(params[0], params[1], &grad[0], &grad[1]);
        
        // Adam更新
        adam_step(&opt, params, grad);
    }
    
    printf("\n训练完成!\n");
    printf("最终位置: (%.6f, %.6f)\n", params[0], params[1]);
    printf("最终损失: %.10f\n", rosenbrock(params[0], params[1]));
    
    return 0;
}
```

**编译运行：**
```bash
gcc adam_optimizer.c -o adam_optimizer -lm
./adam_optimizer
```

## 十、Java实现：面向对象的优化器框架

```java
public class OptimizerFramework {
    
    // 参数向量类
    public static class Vector {
        double[] data;
        
        public Vector(int size) {
            data = new double[size];
        }
        
        public Vector(double... values) {
            data = values.clone();
        }
        
        public int size() { return data.length; }
        
        public Vector add(Vector other) {
            Vector result = new Vector(size());
            for (int i = 0; i < size(); i++) {
                result.data[i] = this.data[i] + other.data[i];
            }
            return result;
        }
        
        public Vector subtract(Vector other) {
            Vector result = new Vector(size());
            for (int i = 0; i < size(); i++) {
                result.data[i] = this.data[i] - other.data[i];
            }
            return result;
        }
        
        public Vector multiply(double scalar) {
            Vector result = new Vector(size());
            for (int i = 0; i < size(); i++) {
                result.data[i] = this.data[i] * scalar;
            }
            return result;
        }
        
        public Vector elementWiseMultiply(Vector other) {
            Vector result = new Vector(size());
            for (int i = 0; i < size(); i++) {
                result.data[i] = this.data[i] * other.data[i];
            }
            return result;
        }
        
        public Vector sqrt() {
            Vector result = new Vector(size());
            for (int i = 0; i < size(); i++) {
                result.data[i] = Math.sqrt(this.data[i]);
            }
            return result;
        }
        
        public Vector square() {
            return elementWiseMultiply(this);
        }
        
        public double get(int i) { return data[i]; }
        public void set(int i, double val) { data[i] = val; }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < data.length; i++) {
                sb.append(String.format("%.4f", data[i]));
                if (i < data.length - 1) sb.append(", ");
            }
            sb.append("]");
            return sb.toString();
        }
    }
    
    // 优化器接口
    public interface Optimizer {
        Vector step(Vector params, Vector grad);
        void reset();
    }
    
    // SGD实现
    public static class SGD implements Optimizer {
        private double lr;
        
        public SGD(double lr) {
            this.lr = lr;
        }
        
        @Override
        public Vector step(Vector params, Vector grad) {
            return params.subtract(grad.multiply(lr));
        }
        
        @Override
        public void reset() {}
    }
    
    // Momentum实现
    public static class Momentum implements Optimizer {
        private double lr;
        private double gamma;
        private Vector velocity;
        
        public Momentum(double lr, double gamma) {
            this.lr = lr;
            this.gamma = gamma;
        }
        
        @Override
        public Vector step(Vector params, Vector grad) {
            if (velocity == null) {
                velocity = new Vector(params.size());
            }
            velocity = velocity.multiply(gamma).add(grad.multiply(lr));
            return params.subtract(velocity);
        }
        
        @Override
        public void reset() {
            velocity = null;
        }
    }
    
    // Adam实现
    public static class Adam implements Optimizer {
        private double lr;
        private double beta1;
        private double beta2;
        private double epsilon;
        private Vector m;
        private Vector v;
        private int t;
        
        public Adam(double lr, double beta1, double beta2, double epsilon) {
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.t = 0;
        }
        
        @Override
        public Vector step(Vector params, Vector grad) {
            t++;
            
            if (m == null) {
                m = new Vector(params.size());
                v = new Vector(params.size());
            }
            
            // 更新一阶矩和二阶矩
            m = m.multiply(beta1).add(grad.multiply(1 - beta1));
            v = v.multiply(beta2).add(grad.square().multiply(1 - beta2));
            
            // 偏差修正
            Vector mHat = m.multiply(1.0 / (1 - Math.pow(beta1, t)));
            Vector vHat = v.multiply(1.0 / (1 - Math.pow(beta2, t)));
            
            // 计算自适应学习率
            Vector denom = vHat.sqrt();
            for (int i = 0; i < denom.size(); i++) {
                denom.set(i, denom.get(i) + epsilon);
            }
            
            // 更新参数
            Vector update = new Vector(params.size());
            for (int i = 0; i < params.size(); i++) {
                update.set(i, (lr * mHat.get(i)) / denom.get(i));
            }
            
            return params.subtract(update);
        }
        
        @Override
        public void reset() {
            m = null;
            v = null;
            t = 0;
        }
    }
    
    // 测试：Rosenbrock函数优化
    public static class Rosenbrock {
        public static double evaluate(Vector x) {
            double a = 1.0, b = 100.0;
            double x1 = x.get(0), x2 = x.get(1);
            return Math.pow(a - x1, 2) + b * Math.pow(x2 - x1 * x1, 2);
        }
        
        public static Vector gradient(Vector x) {
            double a = 1.0, b = 100.0;
            double x1 = x.get(0), x2 = x.get(1);
            
            double dx1 = -2 * (a - x1) - 4 * b * x1 * (x2 - x1 * x1);
            double dx2 = 2 * b * (x2 - x1 * x1);
            
            return new Vector(dx1, dx2);
        }
    }
    
    // 训练函数
    public static void train(Optimizer optimizer, Vector initParams, 
                            int steps, String name) {
        Vector params = new Vector(initParams.data);
        optimizer.reset();
        
        System.out.println("\n" + "=".repeat(50));
        System.out.println("优化器: " + name);
        System.out.println("=".repeat(50));
        System.out.printf("%-10s %-15s %-15s %-15s%n", 
                         "Step", "x1", "x2", "Loss");
        System.out.println("-".repeat(55));
        
        for (int step = 0; step <= steps; step++) {
            double loss = Rosenbrock.evaluate(params);
            Vector grad = Rosenbrock.gradient(params);
            
            if (step % 200 == 0) {
                System.out.printf("%-10d %-15.6f %-15.6f %-15.6f%n",
                    step, params.get(0), params.get(1), loss);
            }
            
            params = optimizer.step(params, grad);
        }
        
        System.out.println("最终参数: " + params);
        System.out.printf("最终损失: %.10f%n", Rosenbrock.evaluate(params));
    }
    
    public static void main(String[] args) {
        Vector initParams = new Vector(-1.0, 2.0);
        
        System.out.println("Rosenbrock函数优化对比");
        System.out.println("初始参数: " + initParams);
        System.out.println("目标参数: [1.0000, 1.0000]");
        System.out.println("初始损失: " + Rosenbrock.evaluate(initParams));
        
        // 对比不同优化器
        train(new SGD(0.001), initParams, 2000, "SGD (lr=0.001)");
        train(new Momentum(0.001, 0.9), initParams, 2000, "Momentum (lr=0.001)");
        train(new Adam(0.01, 0.9, 0.999, 1e-8), initParams, 2000, "Adam (lr=0.01)");
    }
}
```

## 十一、实践建议：如何选择优化器？

### 11.1 快速决策流程

```
开始
  │
  ▼
需要快速实验？ ──是──→ Adam (lr=0.001)
  │否
  ▼
数据量小且干净？ ──是──→ 尝试SGD + Momentum
  │否
  ▼
稀疏数据（如NLP）？ ──是──→ AdaGrad或Adam
  │否
  ▼
需要极致性能？ ──是──→ 精心调参的SGD + Momentum + 学习率调度
  │否
  ▼
默认选择：Adam
```

### 11.2 各优化器适用场景

| 优化器 | 最佳场景 | 避免场景 |
|:---|:---|:---|
| **SGD** | 大规模数据，需要精细调参 | 小规模数据，快速实验 |
| **Momentum** | 存在峡谷地形的问题 | 噪声极大的数据 |
| **AdaGrad** | 稀疏特征（如词袋模型） | 密集特征，长期训练 |
| **RMSProp** | RNN训练 | 需要强动量的问题 |
| **Adam** | 通用首选，快速原型 | 需要极致泛化性能时 |

### 11.3 学习率设置经验

| 优化器 | 推荐初始学习率 | 衰减策略 |
|:---|:---:|:---|
| SGD | 0.01 ~ 0.1 | Step decay: 每30轮×0.1 |
| Momentum | 0.001 ~ 0.01 | 同上 |
| AdaGrad | 0.01 ~ 0.1 | 通常不需要 |
| RMSProp | 0.001 ~ 0.01 | 指数衰减 |
| Adam | 0.0001 ~ 0.001 | 余弦退火或Warmup |

## 十二、总结与展望

### 12.1 优化算法进化史

```
1940s  梯度下降（Cauchy）
  ↓
1964   动量法（Polyak）
  ↓
2011   AdaGrad（Duchi）
  ↓
2012   RMSProp（Hinton）
  ↓
2014   Adam（Kingma & Ba）← 当前默认选择
  ↓
2017   各种改进（AdamW, AMSGrad, RAdam...）
```

### 12.2 核心公式对比

| 算法 | 更新公式 | 核心思想 |
|:---|:---|:---|
| SGD | $\mathbf{w} -= \eta \mathbf{g}$ | 沿梯度方向走 |
| Momentum | $\mathbf{v} = \gamma\mathbf{v} + \eta\mathbf{g}$ | 加惯性 |
| AdaGrad | $\mathbf{r} += \mathbf{g}^2$ | 自适应学习率 |
| RMSProp | $\mathbf{r} = \rho\mathbf{r} + (1-\rho)\mathbf{g}^2$ | 遗忘旧梯度 |
| Adam | $\mathbf{m}, \mathbf{v}$ + 偏差修正 | 动量+自适应 |

### 12.3 关键洞察

1. **没有最好的优化器，只有最适合的优化器**
2. **Adam是安全的默认选择**，但SGD+动量+精心调参往往能达到更好的最终性能
3. **学习率比优化器选择更重要**，花80%时间调学习率，20%时间选优化器
4. **批量大小和学习率相关**，大批量需要更大学习率

**下一篇预告：《第5篇：防止过拟合——正则化技术与Dropout》**

我们将解决深度学习的另一个核心问题：
- L1/L2正则化的原理与实现
- Dropout：随机失活的魔法
- Early Stopping与模型选择
- 数据增强策略

这些都是让模型从"训练集表现好"到"测试集表现好"的关键技术。

---

**本文代码已开源**：[深入探索深度学习的五大核心神经网络架(In depth exploration of the five core neural network architectures of deep learning)](https://github.com/city25/IDEOTFCNNARODL)

**配图清单：**
- 图1：三种梯度下降策略对比图
- 图2：优化算法在损失曲面上的轨迹对比
- 图3：学习率调度策略曲线图
- 图4：Adam与其他优化器收敛速度对比
- 图5：Rosenbrock函数优化轨迹可视化（Python输出）

---

*全文约5600字，涵盖70年优化算法发展史、数学推导、三种语言实现和实用建议。建议读者运行对比实验，直观感受不同优化器的差异，这是选择合适优化器的最佳方式。*

---
> 本文部分内容由AI编辑，可能会出现幻觉，请谨慎阅读。