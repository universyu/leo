# Stackelberg博弈与k-RAND权重优化公式与参数表

## 一、公式汇总

### 1. Stackelberg博弈模型

#### 1.1 博弈参与方定义

$$\text{领导者（防御者）}: \quad \mathbf{w} = (w_{kSP},\; w_{kDG},\; w_{kDS},\; w_{kLO})$$

$$\text{跟随者（攻击者）}: \quad \text{attack} = \text{最优DDoS攻击方案}$$

- 防御者首先选择权重向量 $\mathbf{w}$（算法选择概率分布）
- 攻击者观察 $\mathbf{w}$ 后选择最优攻击策略

#### 1.2 最大最小（Maximin）优化问题

$$\max_{\mathbf{w}} \; \min_{\text{attack}} \; \text{P5\_throughput}(\mathbf{w}, \text{attack})$$

约束条件：

$$w_i \geq w_{min} = 0.05, \quad \forall \, i \in \{kSP, kDG, kDS, kLO\}$$

$$\sum_{i} w_i = 1$$

其中 $w_{min} = 0.05$ 为最低多样性约束，确保每种算法至少有 5% 的被选概率。

---

### 2. k-RAND路由决策

#### 2.1 算法随机选择

对每个数据包的路由决策，生成随机变量 $A$，按概率分布选择算法：

$$A \sim \text{Categorical}(w_{kSP},\; w_{kDG},\; w_{kDS},\; w_{kLO})$$

$$\text{Router}(A) = \begin{cases} \text{k-SP}, & \text{以概率} \; w_{kSP} \\ \text{k-DG}, & \text{以概率} \; w_{kDG} \\ \text{k-DS}, & \text{以概率} \; w_{kDS} \\ \text{k-LO}, & \text{以概率} \; w_{kLO} \end{cases}$$

实现上通过 `numpy.rng.choice(algo_names, p=algo_probs)` 完成。

#### 2.2 有效攻击成本

$$C_{eff}(\mathbf{w}) = \frac{B_{ISL}}{\sum_{i} w_i \cdot p_i}$$

其中 $B_{ISL}$ 为目标ISL链路带宽，$p_i$ 为算法 $i$ 的路径命中目标ISL的概率。权重分布越均匀、各算法路径越分散，$C_{eff}$ 越高，攻击成本越大。

---

### 3. 第一阶段：解析模型快速优化

#### 3.1 有效攻击成本（Component 1）

$$C_{eff}(\mathbf{w}) = \frac{B_{ISL}}{\sum_{a \in \mathcal{A}} w_a \cdot p_a}$$

评分：

$$S_{cost}(\mathbf{w}) = \ln(C_{eff}(\mathbf{w}) + 1)$$

其中 $\mathcal{A} = \{kSP, kDG, kDS, kLO\}$，$p_a$ 为算法 $a$ 的路径命中概率，取对数防止数值主导。

#### 3.2 路径集中度（Component 2）

加权集中度度量 $D(\mathbf{w})$（值越低表示越分散，越好）：

$$D(\mathbf{w}) = \frac{\sum_{a \in \mathcal{A}} w_a \cdot p_a^2}{\sum_{a \in \mathcal{A}} w_a \cdot p_a}$$

评分：

$$S_{div}(\mathbf{w}) = -D(\mathbf{w}) \times 1000$$

#### 3.3 逐地面站攻击成本（Component 3）

$$C_{gs}(\mathbf{w}) = \sum_{gs} \max_{a \in \mathcal{A}} \left[ w_a \cdot C_a(gs) \right]$$

其中 $C_a(gs)$ 为从地面站 $gs$ 通过算法 $a$ 攻击目标ISL所需的带宽成本。

评分：

$$S_{gs}(\mathbf{w}) = \ln(C_{gs}(\mathbf{w}) + 1)$$

#### 3.4 综合目标函数

$$J(\mathbf{w}) = 0.4 \cdot S_{cost}(\mathbf{w}) + 0.3 \cdot S_{div}(\mathbf{w}) + 0.3 \cdot S_{gs}(\mathbf{w})$$

优化目标为最大化 $J(\mathbf{w})$（代码中取负值传入 `scipy.minimize`）。

#### 3.5 三种优化方法

**网格搜索**：在满足 $w_i \geq w_{min}$ 的4维单纯形上均匀采样：

$$w_i = w_{min} + (1 - 4 \cdot w_{min}) \cdot \frac{n_i}{N}, \quad n_i \in \{0, 1, \ldots, N\}$$

**SLSQP约束优化**：以网格搜索最优解为初始点，在以下约束下运行：

$$\text{bounds}: \; w_i \in [w_{min}, \; 1 - 3 \cdot w_{min}], \quad \text{equality}: \; \sum w_i = 1$$

**差分进化**：全局优化，映射3维单位立方体到4维单纯形：

$$w_i = w_{min} + (1 - 4 \cdot w_{min}) \cdot x_i, \quad i = 1,2,3; \quad w_4 = 1 - w_1 - w_2 - w_3$$

三种方法取 $J(\mathbf{w})$ 最大者作为第一阶段结果。

---

### 4. 第二阶段：贝叶斯优化与实际仿真

#### 4.1 高斯过程（GP）代理模型

**RBF核函数**：

$$k(\mathbf{x}, \mathbf{x}') = \exp\left( -\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2} \right)$$

其中 $\ell$ 为长度尺度参数（默认 0.2）。

**核矩阵**：

$$\mathbf{K} = [k(\mathbf{x}_i, \mathbf{x}_j)]_{n \times n} + \sigma_n^2 \mathbf{I}$$

其中 $\sigma_n^2$ 为噪声方差（默认 $10^{-3}$）。

**GP后验预测**：

$$\mu(\mathbf{x}_*) = \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{y}$$

$$\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{k}_*$$

其中 $\mathbf{k}_* = [k(\mathbf{x}_*, \mathbf{x}_i)]_{n \times 1}$，$\mathbf{y} = [y_1, \ldots, y_n]^T$ 为观测值向量。

#### 4.2 期望改进（EI）采集函数

$$\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - y_{best} - \xi) \cdot \Phi(Z) + \sigma(\mathbf{x}) \cdot \phi(Z)$$

其中：

$$Z = \frac{\mu(\mathbf{x}) - y_{best} - \xi}{\sigma(\mathbf{x})}$$

- $\Phi(\cdot)$ 和 $\phi(\cdot)$ 分别为标准正态分布的CDF和PDF
- $y_{best}$ 为当前最优观测值
- $\xi = 0.01$ 为探索-开发平衡参数

#### 4.3 权重空间映射（4维单纯形 → 3维单位立方体）

$$w_i = w_{min} + (1 - 4 \cdot w_{min}) \cdot x_i, \quad i = 1, 2, 3$$

$$w_4 = 1 - w_1 - w_2 - w_3$$

约束 $w_4 \geq w_{min}$，若违反则重新缩放。

#### 4.4 攻击者最优响应模型

攻击者对每个地面站的缩放攻击成本：

$$C_{attack}(gs) = \max_{a \in \mathcal{A}} \left[ C_a(gs) \cdot \frac{w_a}{0.25} \right]$$

攻击总预算约束：

$$C_{total} \leq C_{original} \times 1.5$$

其中 $C_{original}$ 为等权重下的原始攻击成本，$1.5$ 倍系数允许攻击者50%的策略适应裕度。

---

### 5. 第三阶段：最终验证

对比等权重（$w_i = 0.25$）和优化权重的全维度性能差异：

$$\Delta \text{P5} = \text{P5}_{optimized} - \text{P5}_{equal}$$

---

### 6. 最终优化权重

$$\mathbf{w}^* = (w_{kSP}^* = 0.2318, \; w_{kDG}^* = 0.4937, \; w_{kDS}^* = 0.1011, \; w_{kLO}^* = 0.1735)$$

---

## 二、参数表

### 表1：Stackelberg博弈模型参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $\mathbf{w}$ | 权重向量 (weights) | 优化后见表6 | Dict | 四种算法的选择概率分布 |
| $w_{min}$ | 最低权重约束 (MIN_WEIGHT) | 0.05 | float | 每种算法至少5%概率 |
| — | 领导者 | 防御者 | — | 选择路由权重分配 |
| — | 跟随者 | 攻击者 | — | 观察权重后选择最优攻击方案 |
| — | 均衡解目标 | max-min P5吞吐量 | — | 防御者在最优攻击下的最佳防御效果 |

### 表2：k-RAND路由器参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $K$ | 子算法路径数 (k) | 3 | int | 每个子路由器的K值 |
| $w_{kSP}$ | k-SP选择概率 | 0.2318 | float | 优化后的k-SP权重 |
| $w_{kDG}$ | k-DG选择概率 | 0.4937 | float | 优化后的k-DG权重（最高） |
| $w_{kDS}$ | k-DS选择概率 | 0.1011 | float | 优化后的k-DS权重（最低） |
| $w_{kLO}$ | k-LO选择概率 | 0.1735 | float | 优化后的k-LO权重 |
| — | 随机数生成器 | numpy.default_rng | — | 通过seed参数控制可复现性 |
| — | 选择方法 | rng.choice | — | 按概率数组随机选择算法 |

### 表3：第一阶段（解析模型）参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $\alpha_{cost}$ | 攻击成本权重 | 0.4 | float | $S_{cost}$ 在综合目标函数中的权重 |
| $\alpha_{div}$ | 路径集中度权重 | 0.3 | float | $S_{div}$ 在综合目标函数中的权重 |
| $\alpha_{gs}$ | 逐站成本权重 | 0.3 | float | $S_{gs}$ 在综合目标函数中的权重 |
| $N$ | 网格分辨率 (PHASE1_GRID_POINTS) | — | int | 单纯形上的网格采样点数 |
| — | SLSQP最大迭代 | 500 | int | 约束优化最大迭代次数 |
| — | 差分进化最大迭代 | 200 | int | 全局优化最大代数 |
| $B_{ISL}$ | 目标ISL带宽 (ISL_BW) | — | Mbps | 用于有效攻击成本计算 |
| $p_a$ | 算法路径命中概率 | 按统计数据 | float | 各算法路径经过目标ISL的概率 |

### 表4：第二阶段（贝叶斯优化）参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $\ell$ | RBF核长度尺度 (length_scale) | 0.2 | float | 控制核函数的平滑程度 |
| $\sigma_n^2$ | 噪声方差 (noise) | $10^{-3}$ | float | GP模型的观测噪声 |
| $\xi$ | EI探索参数 (xi) | 0.01 | float | 平衡探索与开发的权衡 |
| — | 候选点数 (n_candidates) | 2000 | int | 每轮EI搜索的候选权重数 |
| — | 优化维度 | 3 | int | 4维单纯形映射到3维立方体 |
| — | 攻击预算倍率 | 1.5 | float | 攻击总预算为等权重成本的1.5倍 |
| — | 目标函数 | P5吞吐量 (pps) | — | 通过实际DDoS仿真评估 |

### 表5：三阶段优化策略对比

| 阶段 | 方法 | 目标函数 | 优化器 | 评估速度 | 精度 |
|:---:|:---|:---|:---|:---:|:---:|
| 第一阶段 | 解析模型 | $J(\mathbf{w})$（加权代理评分） | 网格搜索 + SLSQP + 差分进化 | 极快（秒级） | 近似 |
| 第二阶段 | 贝叶斯优化 | 实际P5吞吐量（pps） | GP代理 + EI采集函数 | 慢（分钟级/轮） | 精确 |
| 第三阶段 | 最终验证 | 全维度性能对比 | — | 中等 | 精确 |

### 表6：最终优化权重与安全意义

| 算法 | 优化权重 $w_i^*$ | 占比 | 安全性意义 |
|:---:|:---:|:---:|:---|
| k-DG | 0.4937 | 49.37% | 最高权重：地理多样性路径使攻击者难以通过集中攻击某一区域阻断所有路径 |
| k-SP | 0.2318 | 23.18% | 第二高：路径质量最优，部分路径天然不经过攻击目标链路 |
| k-LO | 0.1735 | 17.35% | 第三高：负载感知能力在攻击进行时提供自适应保护 |
| k-DS | 0.1011 | 10.11% | 最低权重：路径选择与k-DG有较高重叠，k-DG已"替代"部分功能 |

### 表7：综合目标函数各分量

| 分量 | 公式 | 权重 | 意义 |
|:---:|:---|:---:|:---|
| $S_{cost}$ | $\ln(C_{eff}(\mathbf{w}) + 1)$ | 0.4 | 有效攻击成本（越高越难攻击） |
| $S_{div}$ | $-D(\mathbf{w}) \times 1000$ | 0.3 | 路径集中度负值（越低越分散） |
| $S_{gs}$ | $\ln(C_{gs}(\mathbf{w}) + 1)$ | 0.3 | 逐地面站攻击成本（越高越难攻击） |

### 表8：GP代理模型核心公式

| 名称 | 公式 | 说明 |
|:---|:---|:---|
| RBF核函数 | $k(\mathbf{x}, \mathbf{x}') = \exp(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\ell^2})$ | 度量权重向量相似度 |
| GP后验均值 | $\mu(\mathbf{x}_*) = \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{y}$ | 预测未观测点的目标值 |
| GP后验方差 | $\sigma^2(\mathbf{x}_*) = k_{**} - \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{k}_*$ | 预测不确定性 |
| EI采集函数 | $\text{EI}(\mathbf{x}) = (\mu - y_{best} - \xi)\Phi(Z) + \sigma \cdot \phi(Z)$ | 决定下一组评估权重 |
