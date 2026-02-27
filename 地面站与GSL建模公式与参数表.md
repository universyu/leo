# 地面站部署、星地链路与图模型建模公式与参数表

## 一、公式汇总

### 1. 地面站坐标到三维笛卡尔坐标的转换

地面站位于地球表面，令其纬度为 $\varphi_{gs}$，经度为 $\lambda_{gs}$，则三维笛卡尔坐标为：

$$X_{gs} = R_E \cos\varphi_{gs} \cos\lambda_{gs}$$

$$Y_{gs} = R_E \cos\varphi_{gs} \sin\lambda_{gs}$$

$$Z_{gs} = R_E \sin\varphi_{gs}$$

其中 $R_E$ 为地球半径（6,371 km）。

---

### 2. 卫星三维笛卡尔坐标（用于GSL距离计算）

令卫星纬度为 $\varphi_{sat}$，经度为 $\lambda_{sat}$，轨道高度为 $h$，则：

$$r_{sat} = R_E + h$$

$$X_{sat} = r_{sat} \cos\varphi_{sat} \cos\lambda_{sat}$$

$$Y_{sat} = r_{sat} \cos\varphi_{sat} \sin\lambda_{sat}$$

$$Z_{sat} = r_{sat} \sin\varphi_{sat}$$

---

### 3. 地面站与卫星之间的欧几里得距离

$$d_{gs} = \sqrt{(X_{gs} - X_{sat})^2 + (Y_{gs} - Y_{sat})^2 + (Z_{gs} - Z_{sat})^2}$$

---

### 4. 大圆角距（Great Circle Angular Distance）

地面站与卫星星下点之间的角距 $\delta$ 通过球面余弦公式计算：

$$\cos\delta = \sin\varphi_{gs} \sin\varphi_{sat} + \cos\varphi_{gs} \cos\varphi_{sat} \cos(\lambda_{sat} - \lambda_{gs})$$

$$\delta = \arccos(\cos\delta)$$

其中 $\varphi_{gs}, \varphi_{sat}$ 为纬度（弧度），$\lambda_{gs}, \lambda_{sat}$ 为经度（弧度）。为保证数值稳定性，$\cos\delta$ 被限定在 $[-1, 1]$ 范围内。

---

### 5. 仰角（Elevation Angle）计算

仰角 $\varepsilon$ 用于判断卫星对地面站是否可见：

$$\sin\varepsilon = \frac{r_{sat}\cos\delta - R_E}{\sqrt{R_E^2 + r_{sat}^2 - 2R_E \cdot r_{sat}\cos\delta}}$$

其中 $r_{sat} = R_E + h$ 为轨道半径。GSL建立的条件为：

$$\varepsilon \geq \varepsilon_{\min}$$

即仰角须大于等于最低仰角阈值（默认 $\varepsilon_{\min} = 25°$）。

---

### 6. GSL传播时延

$$\tau_{gsl} = \frac{d_{gs}}{c}$$

其中 $d_{gs}$ 为地面站与卫星之间的欧几里得距离，$c$ 为光速（代码中取 299.792 km/ms）。

---

### 7. 回退策略（Fallback）

当可见卫星数量 $N_{visible}$ 不足最低连接数 $N_{min}$ 时：

$$\text{若} \quad N_{visible} < N_{min}, \quad \text{则按} \; d_{gs} \; \text{升序连接最近的卫星，直至连接数} \geq N_{min}$$

默认 $N_{min} = 3$。

---

### 8. NetworkX有向图模型

网络拓扑 $G = (V, E)$ 为有向图（DiGraph），其中：

$$V = V_{sat} \cup V_{gs}$$

$$E = E_{ISL} \cup E_{GSL}$$

- $V_{sat}$：卫星节点集合，每个节点携带属性 $\{type, plane, sat\_idx, position\}$
- $V_{gs}$：地面站节点集合，每个节点携带属性 $\{type, position\}$
- $E_{ISL}$：星间链路有向边集合
- $E_{GSL}$：星地链路有向边集合

每条有向边 $e \in E$ 携带属性：

$$e = (link\_id, \; weight = \tau, \; bandwidth = B, \; link\_type)$$

其中 $\tau$ 为传播时延（作为边权重），$B$ 为带宽。

---

## 二、参数表

### 表1：地面站部署参数

| 参数符号 | 参数名称 | 默认值 | 单位 | 说明 |
|:---:|:---|:---:|:---:|:---|
| — | 地面站数量 | ≈40 | 个 | 覆盖全球主要城市和地区 |
| — | 覆盖区域 | — | — | 北美、欧洲、亚洲、南美、非洲、大洋洲、中东 |
| $\varphi_{gs}$ | 地面站纬度 (latitude) | 参考真实城市 | ° | 各地面站经纬度参考真实地理位置 |
| $\lambda_{gs}$ | 地面站经度 (longitude) | 参考真实城市 | ° | 同上 |

### 表2：星地链路（GSL）属性参数

| 参数符号 | 参数名称 | 默认值 | 单位 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $B_{gsl}$ | GSL带宽 (gsl_bandwidth_mbps) | 1,000 | Mbps (1 Gbps) | 星地链路默认带宽 |
| $\varepsilon_{\min}$ | 最低仰角阈值 (min_elevation_deg) | 25 | ° | 仰角须大于此值才建立GSL |
| $N_{min}$ | 最低连接数 (min_connections) | 3 | 条 | 每个地面站至少连接的卫星数 |
| $\tau_{gsl}$ | GSL传播时延 (propagation_delay) | 根据距离计算 | ms | $\tau_{gsl} = d_{gs} / c$ |
| — | 链路方向 | 双向 | — | 每条GSL自动创建上行和下行两条有向边 |

### 表3：地面站节点属性

| 参数名称 | 类型 | 说明 |
|:---|:---:|:---|
| id | str | 地面站唯一标识符（如城市名缩写） |
| position | (float, float) | 纬度、经度坐标 |
| connected_sats | List[str] | 当前连接的卫星ID列表 |

### 表4：NetworkX有向图边属性

| 属性名称 | 类型 | 说明 |
|:---|:---:|:---|
| link_id | str | 链路唯一标识符（如 GSL_GS_Beijing_SAT_2_5） |
| weight | float | 边权重，取值为传播时延 $\tau$（ms） |
| bandwidth | float | 链路带宽（Mbps） |
| link_type | LinkType | 链路类型（ISL_INTRA / ISL_INTER / GSL） |

### 表5：NetworkX有向图节点属性

| 节点类型 | 属性名称 | 类型 | 说明 |
|:---|:---|:---:|:---|
| 卫星节点 | type | NodeType | SATELLITE |
| 卫星节点 | plane | int | 轨道面编号 |
| 卫星节点 | sat_idx | int | 面内卫星索引 |
| 卫星节点 | position | (float, float, float) | (纬度, 经度, 高度) |
| 地面站节点 | type | NodeType | GROUND_STATION |
| 地面站节点 | position | (float, float) | (纬度, 经度) |

### 表6：物理常数

| 参数符号 | 参数名称 | 取值 | 单位 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $R_E$ | 地球半径 | 6,371 | km | 用于坐标转换与仰角计算 |
| $c$ | 光速 | 299,792 | km/s | 用于GSL传播时延计算（代码中取 299.792 km/ms） |
