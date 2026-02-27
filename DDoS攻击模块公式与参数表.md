# DDoS攻击模块公式与参数表

## 一、公式汇总

### 1. 攻击流速率模型

#### 1.1 洪泛攻击速率分配

每条攻击流的速率通过总速率均分计算：

$$R_{per} = \frac{R_{total}}{N_{attackers}}$$

其中 $R_{total}$ 为总攻击速率（pps），$N_{attackers}$ 为攻击流数量。数据包按泊松到达模式生成：

$$P(k;\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad \lambda = R_{per} \cdot \Delta t$$

其中 $\Delta t$ 为仿真时间步长。

#### 1.2 攻击流实时速率函数

AttackFlow的 `get_current_rate` 方法根据攻击类型和时间返回实时速率，支持三种速率调节机制的叠加：

**（1）渐进式启动（Ramp-up）**：

$$R_{base}(t) = \begin{cases} R_{peak} \cdot \frac{t - t_0}{T_{ramp}}, & t - t_0 < T_{ramp} \\ R_{peak}, & t - t_0 \geq T_{ramp} \end{cases}$$

其中 $T_{ramp}$ 为渐进启动时间，$t_0$ 为攻击开始时间。

**（2）脉冲模式（Pulsing）**：

$$R_{pulse}(t) = \begin{cases} R_{base}(t), & (t - t_0) \bmod T_{cycle} < T_{on} \\ 0, & (t - t_0) \bmod T_{cycle} \geq T_{on} \end{cases}$$

其中 $T_{cycle} = T_{on} + T_{off}$ 为脉冲周期，占空比 $D = T_{on} / T_{cycle}$，默认 $T_{on} = T_{off} = 0.1$s（即200ms周期、50%占空比）。

**（3）Slowloris缩减**：

$$R_{slow}(t) = R_{base}(t) \times \alpha_{slow}, \quad \alpha_{slow} = 0.1$$

#### 1.3 反射放大攻击

为每个攻击者创建两条流：

**触发流**（攻击者→反射器）：

$$R_{trigger} = R_{per}, \quad S_{trigger} = S_{small} = 100 \text{ bytes}$$

**放大流**（反射器→目标）：

$$R_{amplified} = R_{per}, \quad S_{amplified} = S_{small} \times F_{amp}$$

其中 $F_{amp}$ 为放大因子（默认10），流量放大比：

$$\text{放大比} = \frac{S_{amplified}}{S_{trigger}} = F_{amp}$$

#### 1.4 Slowloris攻击总速率

$$R_{total} = N_{attackers} \times C_{per} \times R_{conn}$$

其中 $C_{per}$ 为每个攻击者的连接数，$R_{conn}$ 为每条连接的请求速率。

#### 1.5 协同攻击交错启动

第 $i$ 个攻击向量的启动时间：

$$t_i = t_0 + i \cdot T_{stagger}$$

其中 $T_{stagger}$ 为交错间隔，模拟攻击逐步升级。

---

### 2. 攻击源选择策略

#### 2.1 RANDOM策略

从可用地面站集合 $\mathcal{G}$ 中等概率随机抽样：

$$P(gs_i \text{ 被选}) = \frac{1}{|\mathcal{G}|}, \quad \forall \, gs_i \in \mathcal{G}$$

#### 2.2 DISTRIBUTED策略

将地面站按经度划分为6个区域带，循环轮询各区域：

$$\text{region}(gs_i) = \left\lfloor \frac{\text{lon}(gs_i) + 180}{60} \right\rfloor \bmod 6$$

从6个区域带中循环选取，确保全球均匀分布。

#### 2.3 CLUSTERED策略

以随机种子站 $gs_0$ 为中心，选取满足以下条件的地面站：

$$\mathcal{G}_{cluster} = \{ gs_i \in \mathcal{G} \mid |\text{lon}(gs_i) - \text{lon}(gs_0)| \leq 40° \;\wedge\; |\text{lat}(gs_i) - \text{lat}(gs_0)| \leq 30° \}$$

#### 2.4 PATH_ALIGNED策略

从路由分析已确定路径经过目标ISL的地面站中选取：

$$\mathcal{G}_{aligned} = \{ gs_i \in \mathcal{G} \mid \exists \, path(gs_i, gs_j) \ni e_{target} \}$$

其中 $e_{target}$ 为目标ISL链路，确保100%路径命中率。

---

### 3. 链路定向攻击

#### 3.1 路径穿越检查

对目标ISL链路 $e_{target} = (u, v)$，遍历所有地面站对 $(gs_s, gs_d)$：

$$\text{hit}(gs_s, gs_d) = \begin{cases} 1, & \exists \, i : (path[i] = u \wedge path[i+1] = v) \;\vee\; (path[i] = v \wedge path[i+1] = u) \\ 0, & \text{otherwise} \end{cases}$$

$$\mathcal{P}_{hit} = \{ (gs_s, gs_d, path) \mid \text{hit}(gs_s, gs_d) = 1 \}$$

#### 3.2 攻击流速率均分

从命中对中随机选择 $N_{attackers}$ 个，每条流速率：

$$R_{per} = \frac{R_{total}}{|\mathcal{P}_{selected}|}$$

---

### 4. 脆弱性分析评分

#### 4.1 k-SP评分（路径重叠瓶颈）

对每条ISL链路 $e$，统计其在所有采样对的K条最短路径中的出现次数，以及完全覆盖对数（该链路出现在某对的全部K条路径中）：

$$\text{score}_{kSP}(e) = N_{full}(e) \times 5 + N_{usage}(e)$$

其中 $N_{full}(e)$ 为完全覆盖对数，$N_{usage}(e)$ 为总出现次数，完全覆盖权重为5倍。

#### 4.2 k-DS评分（跨不相交集瓶颈）

统计链路 $e$ 在所有采样对的K条不相交路径中的总出现次数：

$$\text{score}_{kDS}(e) = \sum_{(s,d) \in \mathcal{S}} \sum_{p \in \text{paths}(s,d)} \mathbb{1}[e \in p]$$

该链路是不同通信对的共用瓶颈。

#### 4.3 k-DG评分（面间桥梁瓶颈）

优先统计面间ISL链路（跨轨道面链路）的使用频次：

$$\text{is\_inter}(e) = \begin{cases} 1, & \text{plane}(e.src) \neq \text{plane}(e.dst) \\ 0, & \text{otherwise} \end{cases}$$

$$\text{score}_{kDG}(e) = \begin{cases} N_{inter}(e), & \text{if } \text{is\_inter}(e) = 1 \\ N_{usage}(e), & \text{if no inter-plane links found (fallback)} \end{cases}$$

#### 4.4 k-LO评分（负载级联瓶颈）

综合路径集合覆盖数和总使用次数：

$$\text{score}_{kLO}(e) = N_{pathset}(e) \times 2 + N_{usage}(e)$$

其中 $N_{pathset}(e)$ 为链路 $e$ 出现在多少个不同的（源-目的对的）不相交路径集合中，覆盖数权重为2倍。

---

### 5. 攻击效果度量

#### 5.1 攻击带宽计算

$$BW_{attack}(t) = \sum_{f \in \mathcal{F}_{attack}} R_f(t) \cdot S_f \cdot 8 \times 10^{-6} \quad (\text{Mbps})$$

其中 $R_f(t)$ 为攻击流 $f$ 在时刻 $t$ 的速率（pps），$S_f$ 为数据包大小（bytes）。

#### 5.2 受影响链路判定

$$N_{affected} = |\{ \ell \in \mathcal{L} \mid U(\ell) > 0.8 \}|$$

其中 $U(\ell)$ 为链路 $\ell$ 的利用率，阈值为80%。

#### 5.3 攻击资源成本估算（DDoSAttackGenerator）

$$C_{attack} = C_{host} + C_{bw}$$

**基础成本**：

$$C_{host} = N_{attackers} \times 10, \quad C_{bw} = BW_{attack} \times 0.1$$

**攻击类型修正系数**：

$$\text{反射攻击}: \quad C_{host}' = C_{host} \times 0.5, \quad C_{bw}' = C_{bw} \times F_{amp} \times 0.2$$

$$\text{Slowloris}: \quad C_{host}' = C_{host} \times 0.3, \quad C_{bw}' = C_{bw} \times 0.1$$

#### 5.4 防御效果攻击成本（AttackCostCalculator）

$$\text{AttackCost} = \frac{BW_{attack} \text{ (Mbps)}}{L_{induced}}$$

其中攻击诱导丢包率：

$$L_{induced} = \max(0, \; L_{current} - L_{baseline})$$

- $L_{current}$ 为当前正常数据包丢包率
- $L_{baseline}$ 为无攻击基线丢包率

AttackCost 越高，表示攻击者需要更多流量才能造成同等损害，即防御效果越好。

#### 5.5 归一化攻击成本

$$C_{normalized}(L_{target}) = BW_{attack} \times \frac{L_{target}}{L_{induced}}$$

线性外推达到目标丢包率 $L_{target}$（默认10%）所需的攻击流量。

#### 5.6 攻击效率

$$\eta = \frac{N_{normal\_dropped}}{N_{attack\_sent}}$$

$$\text{DPM} = \frac{L_{induced}}{BW_{attack} \text{ (Mbps)}}$$

其中 $\eta$ 为攻击效率（值越低防御越好），DPM（Damage Per Mbps）为每Mbps攻击流量造成的损害。

---

## 二、参数表

### 表1：攻击类型枚举（AttackType）

| 枚举值 | 名称 | 描述 | 特征 |
|:---:|:---|:---|:---|
| `FLOODING` | 容量耗尽型洪泛 | 高速率大包直接消耗带宽 | 恒定速率，泊松到达 |
| `REFLECTION` | 反射放大 | 小包触发、大包放大 | 放大因子默认10倍 |
| `SLOWLORIS` | 慢速持续 | 低速率持续连接耗尽资源 | 速率缩减因子0.1 |
| `PULSING` | 脉冲攻击 | 周期性开关切换 | 默认200ms周期、50%占空比 |
| `COORDINATED` | 协同多向量 | 同时启动多种攻击向量 | 支持交错启动 |
| `LINK_TARGETED` | 链路定向 | 精准攻击指定ISL链路 | 需完全网络知识 |
| `BOTTLENECK` | 瓶颈攻击 | 攻击网络瓶颈节点 | 基于地理位置分析 |

### 表2：攻击源选择策略（AttackStrategy）

| 枚举值 | 名称 | 选择逻辑 | 适用场景 |
|:---:|:---|:---|:---|
| `RANDOM` | 随机选择 | 等概率随机抽样 | 无先验知识 |
| `DISTRIBUTED` | 全球分布 | 6个经度区域带循环轮询 | 模拟全球僵尸网络 |
| `CLUSTERED` | 区域聚集 | 经度差≤40°、纬度差≤30°范围内 | 模拟区域性僵尸网络 |
| `PATH_ALIGNED` | 路径对齐 | 从经过目标链路的地面站中选取 | 精准攻击，100%命中 |

### 表3：AttackConfig配置参数

| 参数名 | 默认值 | 类型 | 说明 |
|:---|:---:|:---:|:---|
| attack_type | FLOODING | AttackType | 攻击类型 |
| targets | [] | List[str] | 目标节点ID列表 |
| num_attackers | 10 | int | 攻击源数量 |
| total_rate | 5000.0 | float | 总攻击速率（pps） |
| packet_size | 1000 | int | 数据包大小（bytes） |
| start_time | 0.0 | float | 攻击开始时间（s） |
| duration | -1.0 | float | 攻击持续时间（s），-1为无限 |
| strategy | DISTRIBUTED | AttackStrategy | 攻击源选择策略 |
| amplification_factor | 1.0 | float | 反射放大因子 |
| pulse_on_time | 0.1 | float | 脉冲开启时间（s） |
| pulse_off_time | 0.1 | float | 脉冲关闭时间（s） |
| connections_per_attacker | 100 | int | Slowloris每攻击者连接数 |
| request_rate | 1.0 | float | Slowloris每连接请求速率 |
| ramp_up_time | 0.0 | float | 渐进式启动时间（s） |

### 表4：AttackFlow速率参数

| 参数名 | 默认值 | 类型 | 说明 |
|:---|:---:|:---:|:---|
| rate | — | float | 基础速率（pps） |
| pulse_on | 0.1 | float | 脉冲开启时长（s） |
| pulse_off | 0.1 | float | 脉冲关闭时长（s） |
| amplification | 1.0 | float | 放大系数（反射攻击用） |
| ramp_up_time | 0.0 | float | 从0线性增长到峰值的时间 |
| $\alpha_{slow}$ | 0.1 | float | Slowloris速率缩减因子 |
| pattern | POISSON | TrafficPattern | 数据包到达模式 |
| flow_type | ATTACK | PacketType | 数据包类型标记 |

### 表5：脆弱性分析评分逻辑对比

| 路由算法 | 评分策略 | 评分公式 | 攻击原理 |
|:---:|:---|:---|:---|
| k-SP | 路径重叠瓶颈 | $N_{full} \times 5 + N_{usage}$ | K条最短路径高度重叠，攻击一条链路可瘫痪全部备选路径 |
| k-DS | 跨不相交集瓶颈 | $\sum \mathbb{1}[e \in p]$ | 不同通信对的不相交路径集共享同一瓶颈链路 |
| k-DG | 面间桥梁瓶颈 | 优先面间ISL使用频次 | 地理多样性路径仍需跨轨道面桥梁链路 |
| k-LO | 负载级联瓶颈 | $N_{pathset} \times 2 + N_{usage}$ | 攻击触发级联重路由导致所有备选路径拥塞 |

### 表6：攻击资源成本系数

| 攻击类型 | 主机成本系数 | 带宽成本系数 | 说明 |
|:---:|:---:|:---:|:---|
| 洪泛（基础） | $\times 1.0$ | $\times 1.0$ | $C_{host} = N \times 10$，$C_{bw} = BW \times 0.1$ |
| 反射放大 | $\times 0.5$ | $\times F_{amp} \times 0.2$ | 主机成本降半，带宽成本按放大因子调整 |
| Slowloris | $\times 0.3$ | $\times 0.1$ | 主机和带宽成本大幅降低 |

### 表7：AttackMetrics度量指标

| 指标名 | 类型 | 说明 |
|:---|:---:|:---|
| attack_packets_sent | int | 攻击数据包发送总数 |
| attack_packets_delivered | int | 攻击数据包到达目标数 |
| attack_bandwidth_used | float (Mbps) | 攻击消耗的总带宽 |
| normal_packets_affected | int | 受攻击影响的正常数据包数 |
| attack_cost | float | 估算的攻击资源成本 |
| attack_efficiency | float | 攻击效率（损害/成本） |
| max_link_utilization | float | 攻击期间最大链路利用率 |
| affected_links | int | 利用率>80%的受影响链路数 |
| bottleneck_events | int | 瓶颈事件计数 |

### 表8：AttackCostCalculator防御效果度量

| 度量名称 | 公式 | 意义 | 越高/越低越好 |
|:---|:---|:---|:---:|
| 攻击成本 (Attack Cost) | $BW_{attack} / L_{induced}$ | 每单位损害所需攻击流量 | 越高 |
| 归一化成本 | $BW_{attack} \times (L_{target} / L_{induced})$ | 达到10%丢包率所需流量 | 越高 |
| 攻击效率 | $N_{dropped} / N_{attack\_sent}$ | 每个攻击包造成的正常包丢弃 | 越低 |
| 每Mbps损害 (DPM) | $L_{induced} / BW_{attack}$ | 每Mbps攻击流量造成的丢包率 | 越低 |

### 表9：各攻击类型默认参数汇总

| 攻击类型 | num_attackers | total_rate (pps) | packet_size (B) | 特殊参数 |
|:---:|:---:|:---:|:---:|:---|
| 洪泛 | 10 | 5000 | 1000 | — |
| 反射放大 | 10 | 5000 | 100（触发）/1000（放大） | amplification_factor=10 |
| 脉冲 | 20 | peak_rate | 1000 | on=0.1s, off=0.1s |
| Slowloris | 20 | $N \times C \times R$ | 500 | connections=100, rate=1.0 |
| 协同 | 按向量分配 | 按向量均分 | — | stagger_time, attack_vectors |
| 链路定向 | 按命中对数 | 均分到命中对 | 1000 | target_link, router |
