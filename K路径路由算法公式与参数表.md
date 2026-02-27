# K路径路由算法公式与参数表

## 一、公式汇总

### 1. k-SP：K最短路径算法

#### 1.1 路径权重（传播时延）

路径 $P = (v_0, v_1, \ldots, v_n)$ 的总权重为各链路传播时延之和：

$$W(P) = \sum_{i=0}^{n-1} \tau(v_i, v_{i+1})$$

其中 $\tau(v_i, v_{i+1})$ 为链路 $(v_i, v_{i+1})$ 的传播时延。

#### 1.2 K最短简单路径枚举（Yen's算法）

在加权有向图 $G=(V,E)$ 中，求从源节点 $s$ 到目的节点 $d$ 按权重排序的前 $K$ 条简单路径：

$$\{P_1, P_2, \ldots, P_K\} = \underset{P \in \mathcal{P}(s,d)}{\text{K-argmin}} \; W(P)$$

其中 $\mathcal{P}(s,d)$ 为所有从 $s$ 到 $d$ 的简单路径集合，满足 $W(P_1) \leq W(P_2) \leq \cdots \leq W(P_K)$。

#### 1.3 路径缓存

使用字典 `path_cache` 以 $(s, d)$ 元组为键缓存结果：

$$\text{path\_cache}[(s, d)] = \{P_1, P_2, \ldots, P_K\}$$

首次查询执行完整计算，后续查询时间复杂度为 $O(1)$。

#### 1.4 路径选择

k-SP 的 `compute_path` 返回第一条（即最短）路径：

$$P^* = P_1 = \arg\min_{P \in \mathcal{P}(s,d)} W(P)$$

---

### 2. k-DS：K不相交最短路径算法

#### 2.1 链路不相交性定义

路径集合 $\{P_1, P_2, \ldots, P_K\}$ 满足链路不相交：

$$\forall \, i \neq j: \quad E(P_i) \cap E(P_j) = \emptyset$$

其中 $E(P_i) = \{(v_l, v_{l+1}) \mid v_l, v_{l+1} \in P_i\}$ 为路径 $P_i$ 上的链路集合。

#### 2.2 节点不相交性定义

路径集合 $\{P_1, P_2, \ldots, P_K\}$ 满足节点不相交（源和目的除外）：

$$\forall \, i \neq j: \quad V^{\circ}(P_i) \cap V^{\circ}(P_j) = \emptyset$$

其中 $V^{\circ}(P_i) = V(P_i) \setminus \{s, d\}$ 为路径 $P_i$ 的中间节点集合。

#### 2.3 迭代移除算法

设图副本为 $G'$，迭代过程如下：

$$\text{for} \; k = 1, 2, \ldots, K:$$

$$\quad P_k = \text{shortest\_path}(G', s, d)$$

$$\quad \text{链路不相交模式}: \; G' \leftarrow G' \setminus E(P_k)$$

$$\quad \text{节点不相交模式}: \; G' \leftarrow G' \setminus V^{\circ}(P_k)$$

若在第 $k$ 步无法找到路径，则算法终止，返回已获得的 $k-1$ 条路径。

#### 2.4 随机路径选择

从 $K$ 条不相交路径中均匀随机选择一条：

$$P^* = P_j, \quad j \sim \text{Uniform}(\{1, 2, \ldots, K\})$$

---

### 3. k-DG：K不相交地理多样性路径算法

#### 3.1 候选路径生成

生成 $K \times 5$ 条候选路径（5倍冗余）：

$$\mathcal{C} = \{C_1, C_2, \ldots, C_{5K}\} = \underset{P \in \mathcal{P}(s,d)}{5K\text{-argmin}} \; W(P)$$

#### 3.2 链路不相交性检查

维护已用链路集合 $\mathcal{L}_{used}$，候选路径 $C$ 的链路集合为：

$$E'(C) = \{(\min(v_i, v_{i+1}), \max(v_i, v_{i+1})) \mid (v_i, v_{i+1}) \in C\}$$

链路不相交条件：

$$E'(C) \cap \mathcal{L}_{used} = \emptyset$$

通过后更新：$\mathcal{L}_{used} \leftarrow \mathcal{L}_{used} \cup E'(C)$。

#### 3.3 节点多样性

新路径 $P_{new}$ 相对于已选路径 $P_{exist}$ 的节点多样性：

$$D_{node}(P_{new}, P_{exist}) = 1 - \frac{|V^{\circ}(P_{new}) \cap V^{\circ}(P_{exist})|}{\max(|V^{\circ}(P_{new})|, |V^{\circ}(P_{exist})|, 1)}$$

#### 3.4 轨道面多样性

新路径 $P_{new}$ 相对于已选路径 $P_{exist}$ 的轨道面多样性：

$$\Pi(P) = \{\text{plane}(v) \mid v \in V^{\circ}(P),\; v \in V_{sat}\}$$

$$D_{plane}(P_{new}, P_{exist}) = 1 - \frac{|\Pi(P_{new}) \cap \Pi(P_{exist})|}{\max(|\Pi(P_{new})|, |\Pi(P_{exist})|, 1)}$$

#### 3.5 综合地理多样性评分

对每条已选路径求加权平均：

$$D(P_{new}, \{P_1, \ldots, P_m\}) = \frac{1}{m} \sum_{j=1}^{m} \left[ 0.5 \cdot D_{node}(P_{new}, P_j) + 0.5 \cdot D_{plane}(P_{new}, P_j) \right]$$

若无已选路径（$m=0$），则 $D = 1.0$。

#### 3.6 接受条件

候选路径被接受须满足：

$$D(P_{new}, \{P_1, \ldots, P_m\}) > D_{threshold}$$

其中多样性阈值 $D_{threshold} = 0.3$，或者为第一条路径（自动接受）。

---

### 4. k-LO：K链路不相交负载优化路径算法

#### 4.1 路径最大链路利用率

$$U_{max}(P) = \max_{(v_i, v_{i+1}) \in P} \; u(v_i, v_{i+1})$$

其中 $u(v_i, v_{i+1})$ 为链路 $(v_i, v_{i+1})$ 的当前利用率。

#### 4.2 路径平均链路利用率

$$U_{avg}(P) = \frac{1}{|E(P)|} \sum_{(v_i, v_{i+1}) \in P} u(v_i, v_{i+1})$$

其中 $|E(P)|$ 为路径上的链路数量。

#### 4.3 综合评分（分段函数）

$$S(P) = \begin{cases} U_{avg}(P) \times 100 + W(P), & \text{若} \; U_{max}(P) \leq U_{threshold} \\ U_{max}(P) \times 1000 + U_{avg}(P) \times 100 + W(P), & \text{若} \; U_{max}(P) > U_{threshold} \end{cases}$$

其中 $U_{threshold}$ 为负载阈值（默认 0.7），$W(P)$ 为路径传播时延。

#### 4.4 最优路径选择

$$P^* = \arg\min_{P \in \{P_1, \ldots, P_K\}} S(P)$$

当 $U_{max}(P) > U_{threshold}$ 时，1000倍惩罚系数大幅降低高负载路径的选择优先级。

#### 4.5 缓存刷新机制

$$\text{每} \; N_{refresh} \; \text{个数据包清空路径缓存}:$$

$$\text{若} \quad \text{packet\_count} \geq N_{refresh}, \quad \text{则} \; \text{path\_cache} \leftarrow \emptyset, \; \text{packet\_count} \leftarrow 0$$

默认 $N_{refresh} = 100$，在静态拓扑约束下实现准动态负载均衡。

---

## 二、参数表

### 表1：k-SP算法参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $K$ | 路径数量 (k) | 3 | int | 枚举的最短路径条数 |
| — | 权重属性 (weight) | "weight" | str | 用于计算路径权重的边属性（传播时延） |
| — | 路径缓存 (path_cache) | {} | Dict | 以 $(s,d)$ 为键缓存K条最短路径 |
| — | 基础算法 | Yen's算法 | — | 通过 NetworkX `shortest_simple_paths` 实现 |
| — | 截断方式 | `itertools.islice` | — | 生成器模式，取前K条后停止 |

### 表2：k-DS算法参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $K$ | 路径数量 (k) | 3 | int | 不相交路径条数 |
| — | 不相交类型 (disjoint_type) | "link" | str | "link"（链路不相交）或 "node"（节点不相交） |
| — | 路径选择方式 | 均匀随机 | — | 从K条路径中随机选择一条 |
| — | 路径缓存 (path_cache) | {} | Dict | 以 $(s,d)$ 为键缓存K条不相交路径 |
| — | 基础算法 | Dijkstra + 迭代移除 | — | 每轮求最短路后移除对应链路或中间节点 |

### 表3：k-DG算法参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $K$ | 路径数量 (k) | 3 | int | 地理多样性路径条数 |
| — | 多样性权重 (diversity_weight) | 0.5 | float | 地理多样性的权重系数（0-1） |
| $5K$ | 候选路径数量 | $K \times 5$ | int | 5倍冗余确保足够的多样性选择空间 |
| $D_{threshold}$ | 多样性阈值 | 0.3 | float | 候选路径的最低多样性评分 |
| $w_{node}$ | 节点多样性权重 | 0.5 | float | 节点多样性在综合评分中的权重 |
| $w_{plane}$ | 轨道面多样性权重 | 0.5 | float | 轨道面多样性在综合评分中的权重 |
| — | 不相交类型 | 链路不相交 | — | 基于 `used_links` 集合维护链路不相交性 |
| — | 路径缓存 (path_cache) | {} | Dict | 以 $(s,d)$ 为键缓存K条地理多样性路径 |

### 表4：k-LO算法参数

| 参数符号 | 参数名称 | 默认值 | 类型 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $K$ | 路径数量 (k) | 3 | int | 链路不相交路径条数 |
| $U_{threshold}$ | 负载阈值 (load_threshold) | 0.7 | float | 高负载判定阈值，超过时施加惩罚 |
| $N_{refresh}$ | 缓存刷新间隔 (recompute_interval) | 100 | int | 每处理100个数据包清空路径缓存 |
| — | 低负载评分公式 | $U_{avg} \times 100 + W$ | — | 当 $U_{max} \leq 0.7$ 时使用 |
| — | 高负载惩罚系数 | 1000 | — | 当 $U_{max} > 0.7$ 时对 $U_{max}$ 施加1000倍惩罚 |
| — | 底层路由器 | KDSRouter(link) | — | 复用k-DS链路不相交路由器获取K条路径 |
| — | 路径选择方式 | 最低评分 | — | 选择综合评分 $S(P)$ 最小的路径 |

### 表5：四种算法对比总览

| 算法 | 类名 | 路径特性 | 选择策略 | 缓存策略 | 核心优势 |
|:---:|:---|:---|:---|:---|:---|
| k-SP | KShortestPathsRouter | 可重叠的K条最短路径 | 选择第一条（最短） | 静态缓存 | 路径质量最优 |
| k-DS | KDSRouter | 链路/节点不相交路径 | 均匀随机选择 | 静态缓存 | 单链路故障隔离 |
| k-DG | KDGRouter | 链路不相交 + 地理分散 | 贪心多样性筛选 | 静态缓存 | 抵御区域性攻击 |
| k-LO | KLORouter | 链路不相交 + 负载感知 | 最低综合评分 | 每100包刷新 | 动态负载均衡 |

### 表6：评分函数参数明细

| 参数符号 | 含义 | 计算方式 | 说明 |
|:---:|:---|:---|:---|
| $U_{max}(P)$ | 路径最大链路利用率 | $\max_{e \in P} u(e)$ | 衡量路径上最拥塞的链路 |
| $U_{avg}(P)$ | 路径平均链路利用率 | $\frac{1}{\|E(P)\|} \sum_{e \in P} u(e)$ | 衡量路径整体负载水平 |
| $W(P)$ | 路径传播时延 | $\sum_{e \in P} \tau(e)$ | 路径上所有链路时延之和 |
| $S(P)$ | 综合评分 | 分段函数（见公式4.3） | 用于k-LO路径选择 |
| $D(\cdot)$ | 地理多样性评分 | 加权平均（见公式3.5） | 用于k-DG路径筛选 |
