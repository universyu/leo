
# LEO建模公式与参数表

## 一、公式汇总

### 1. Walker星座构型描述

Walker星座通过三元组描述：

$$T / P / F$$

其中 $T$ 为星座总卫星数，$P$ 为轨道平面数，$F$ 为相位因子。本项目取 $T=66, P=6, F=1$。

---

### 2. 升交点赤经（RAAN）

$$\Omega_p = \frac{360°}{P} \times p, \quad p = 0, 1, \ldots, P-1$$

其中 $p$ 为轨道平面编号，$P$ 为轨道平面总数。

---

### 3. 平近点角（Mean Anomaly）

$$M_s = \frac{360°}{S} \times s, \quad s = 0, 1, \ldots, S-1$$

其中 $s$ 为卫星在轨道面内的编号，$S$ 为每个轨道面内的卫星数。

---

### 4. 轨道坐标到地心惯性坐标系的旋转矩阵变换

令纬度幅角 $u = M$（简化模型下平近点角即纬度幅角），轨道面内位置为：

$$x_{\text{orb}} = \cos u, \quad y_{\text{orb}} = \sin u$$

经旋转矩阵投影到地心惯性坐标系（ECI，简化不含地球自转）：

$$x = x_{\text{orb}} \cos\Omega - y_{\text{orb}} \cos i \cdot \sin\Omega$$

$$y = x_{\text{orb}} \sin\Omega + y_{\text{orb}} \cos i \cdot \cos\Omega$$

$$z = y_{\text{orb}} \sin i$$

其中 $\Omega$ 为升交点赤经，$i$ 为轨道倾角。

---

### 5. 地心惯性坐标到地理坐标的转换

$$\text{lat} = \arcsin(z)$$

$$\text{lon} = \text{atan2}(y, x)$$

---

### 6. 地理坐标到三维笛卡尔坐标的转换

$$r = R_E + h$$

$$X = r \cos(\text{lat}) \cos(\text{lon})$$

$$Y = r \cos(\text{lat}) \sin(\text{lon})$$

$$Z = r \sin(\text{lat})$$

其中 $R_E$ 为地球半径，$h$ 为轨道高度。

---

### 7. 两颗卫星之间的欧几里得距离

$$d = \sqrt{(X_1 - X_2)^2 + (Y_1 - Y_2)^2 + (Z_1 - Z_2)^2}$$

---

### 8. ISL传播时延

$$\tau = \frac{d}{c}$$

其中 $c$ 为光速（299,792 km/s），$\tau$ 的单位为毫秒（代码中 $c$ 取 299.792 km/ms）。

---

### 9. 链路利用率

$$U = \frac{L_{\text{current}}}{C_{\text{link}}}$$

其中 $L_{\text{current}}$ 为当前时间步内链路上已传输的比特数，$C_{\text{link}}$ 为链路在当前时间步内的容量（比特），计算为：

$$C_{\text{link}} = B \times 10^6 \times \Delta t$$

其中 $B$ 为带宽（Mbps），$\Delta t$ 为仿真时间步长（秒）。当 $U > 1.0$ 时判定为拥塞。

---

## 二、参数表

### 表1：Walker星座构型参数

| 参数符号 | 参数名称 | 默认值 | 说明 |
|:---:|:---|:---:|:---|
| $P$ | 轨道平面数 (num_planes) | 6 | 等间距分布的轨道平面数量 |
| $S$ | 每面卫星数 (sats_per_plane) | 11 | 每个轨道面内均匀分布的卫星数 |
| $T$ | 总卫星数 | 66 | $T = P \times S$ |
| $h$ | 轨道高度 (altitude_km) | 550 km | LEO轨道距地面高度 |
| $i$ | 轨道倾角 (inclination_deg) | 53° | 轨道面与赤道面的夹角 |
| $F$ | 相位因子 | 1 | Walker构型相位因子 |
| $\Delta\Omega$ | RAAN间距 | 60° | $\Delta\Omega = 360° / P$ |
| $\Delta\nu$ | 卫星角间距 | ≈32.7° | $\Delta\nu = 360° / S$ |

### 表2：卫星节点属性参数

| 参数符号 | 参数名称 | 默认值 | 单位 | 说明 |
|:---:|:---|:---:|:---:|:---|
| — | 唯一标识符 (id) | SAT_{p}_{s} | — | 由轨道面编号 $p$ 和面内编号 $s$ 组成 |
| — | 轨道面编号 (plane_id) | $p$ | — | 所属轨道平面索引 |
| — | 面内编号 (sat_id) | $s$ | — | 轨道面内的卫星索引 |
| — | 三维位置 (position) | (lat, lon, $h$) | (°, °, km) | 纬度、经度、高度 |
| $C_{\text{sat}}$ | 处理容量 (capacity) | 10,000 | packets/s | 卫星每秒可处理的数据包数 |
| $B_{\text{sat}}$ | 缓冲区大小 (buffer_size) | 1,000 | packets | 卫星队列缓冲区容量 |

### 表3：星间链路（ISL）属性参数

| 参数符号 | 参数名称 | 默认值 | 单位 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $B$ | 链路带宽 (bandwidth) | 10,000 | Mbps (10 Gbps) | ISL默认带宽 |
| $\tau$ | 传播时延 (propagation_delay) | 根据距离计算 | ms | 由卫星间欧几里得距离 / 光速得到 |
| $L_{\text{current}}$ | 当前负载 (current_load) | 0 | bits | 当前时间步内已传输的比特数 |
| $\Delta t$ | 仿真时间步长 (time_step) | 0.001 | s (1ms) | 离散仿真的时间步长 |
| $U$ | 链路利用率 | — | — | $U = L_{\text{current}} / C_{\text{link}}$，$U > 1.0$ 时判定拥塞 |

### 表4：物理常数

| 参数符号 | 参数名称 | 取值 | 单位 | 说明 |
|:---:|:---|:---:|:---:|:---|
| $R_E$ | 地球半径 | 6,371 | km | 用于坐标转换 |
| $c$ | 光速 | 299,792 | km/s | 用于传播时延计算（代码中取 299.792 km/ms） |
