# 光路与物理参数基线

> [!IMPORTANT]
> 本文是 `configs/physics_cfg.yaml` 的设定光路基线说明。
> `OpticalPathVAE` 会直接引用本文对应的顺序、距离和读出平面语义，但本文仍然只维护物理平台与参数基线，不展开训练损失或扩散实现细节。

## 1. 文档角色

- 本文用于保留实验平台的真实光路、元件顺序和刚性成像约束。
- 本文用于解释 `physics_cfg.yaml` 中的宏观光路参数和读出参数来自哪里。
- 本文用于为 `OpticalPathVAE` 提供前级光路、后级光路和读出平面的基线资料。
- 本文不承担第一阶段概率瓶颈、扩散训练或 checkpoint 策略说明。

## 2. 当前保留范围

以下内容在迁移到新主线时直接保留并进入第一阶段前向：

- 光路顺序与元件角色
- 平台孔距换算规则：`25.4 mm = 1 inch`
- `4f relay system` 的几何长度约束
- 芯片后 `3 mm` 探测平面约束
- 与 `physics_cfg.yaml` 一一对应的物理参数和距离字段
- `camera` 平面主读出语义

以下内容仍只作为历史背景保留，不再作为当前主线描述：

- `legacy LightGenVAE`
- 基于 latent prior 的生成逻辑
- 通过反向优化 SLM 输入图完成采样的流程
- `proxy_input_optimization` 相关执行链路

## 3. 光路路径

完整光路顺序从激光到相机如下：

`Laser -> Mirror(m1) -> Beam Expander -> Mirror(m2) -> PF1 -> PBS -> BS -> 反射式 SLM -> BS -> Mirror(m3) -> 4f Relay -> PF2 -> LightGen chip -> 芯片后 3 mm 探测平面 -> Relay Lens -> Camera`

对应解释：

- `SLM` 是反射式 LCOS，因此 `PBS` 同时承担入射分离和反射后透射出射。
- `4f relay system` 是整条光路中最刚性的段，长度由焦距决定。
- `LightGen chip` 后不是直接读相机，而是先定义芯片后固定 `3 mm` 的探测平面，再由 relay lens 成像到相机。
- 在 `OpticalPathVAE` 中，前级光路用于 `encode()`，后级 `probe -> relay lens -> camera` 用于 `decode()`。

## 4. 与 `physics_cfg.yaml` 的字段映射

### 4.1 物理常数与读出

| 物理含义 | 配置字段 | 当前值 |
|---|---|---|
| 波长 | `optics.wavelength` | `532.0e-9` |
| 像素尺寸 | `optics.pixel_size` | `3.0e-6` |
| 网格尺寸 | `geometry.grid_size` | `200` |
| 传播步长 | `geometry.prop_dist` | `1.0e-3` |
| 主读出平面 | `readout.primary_plane` | `camera` |
| 相机平面保留 | `readout.save_camera_plane` | `true` |
| 归一化分位数 | `readout.normalize_percentile` | `99.5` |

### 4.2 宏观光路距离

| 段落 | 配置字段 | 当前值 |
|---|---|---|
| Laser -> m1 | `macro_optics.dist_laser_to_m1` | `0.115 m` |
| m1 -> m2 | `macro_optics.dist_m1_to_m2` | `0.215 m` |
| m2 -> PF1 | `macro_optics.dist_m2_to_pf1` | `0.215 m` |
| PF1 -> PBS | `macro_optics.dist_pf1_to_pbs` | `0.115 m` |
| PBS -> BS | `macro_optics.dist_pbs_to_bs` | `0.088 m` |
| BS -> SLM | `macro_optics.dist_bs_to_slm` | `0.038 m` |
| SLM -> m3 | `macro_optics.dist_slm_to_m3` | `0.140 m` |
| m3 -> 4f 入口 | `macro_optics.dist_m3_to_4f` | `0.063 m` |
| 4f 焦距 | `macro_optics.f_4f` | `0.100 m` |
| 4f -> PF2 | `macro_optics.dist_4f_to_pf2` | `0.088 m` |
| chip -> 探测平面 | `macro_optics.dist_chip_to_plane` | `0.003 m` |
| 探测平面 -> relay lens | `macro_optics.dist_plane_to_lens` | `0.105 m` |
| relay lens 焦距 | `macro_optics.f_relay` | `0.075 m` |
| relay lens -> camera | `macro_optics.dist_lens_to_cam` | `0.260 m` |

## 5. 平台孔距换算与距离校验

平台孔距按 `25.4 mm` 为基准换算。下表给出保留的经验距离范围，并对照当前配置值。

| 序号 | 段落 | 孔距范围 | 经验距离 | 当前配置值 | 结论 |
|---|---|---:|---:|---:|---|
| 1 | `m1 -> Beam Expander` | `4-5` | `101.6-127.0 mm` | 归入前段准直空间 | 保留为实验经验 |
| 2 | `Beam Expander -> m2` | `8-9` | `203.2-228.6 mm` | 归入扩束传播空间 | 保留为实验经验 |
| 3 | `m2 -> PF1` | `8-9` | `203.2-228.6 mm` | `215 mm` | 与配置一致 |
| 4 | `PF1 -> PBS` | `4-5` | `101.6-127.0 mm` | `115 mm` | 与配置一致 |
| 5 | `PBS -> BS` | `3-4` | `76.2-101.6 mm` | `88 mm` | 与配置一致 |
| 6 | `BS -> SLM` | `1-2` | `25.4-50.8 mm` | `38 mm` | 与配置一致 |
| 7 | `BS -> m3` | `5-6` | `127.0-152.4 mm` | 由 `SLM -> m3` 侧体现 | 保留为折叠布局经验 |
| 8 | `m3 -> 4f 入口` | `2-3` | `50.8-76.2 mm` | `63 mm` | 与配置一致 |
| 9 | `4f 内部跨度` | `14-15` | `355.6-381.0 mm` | `≈ 4f = 400 mm` 主体成像约束 | 量级一致，受透镜布置影响 |
| 10 | `4f -> PF2` | `3-4` | `76.2-101.6 mm` | `88 mm` | 与配置一致 |
| 11 | `PF2 -> chip` | `2-3` | `50.8-76.2 mm` | 含在 4f 输出到芯片的实验布局中 | 保留为实验经验 |
| 12 | `chip -> relay lens` | `5-6` | `127.0-152.4 mm` | `3 mm + 105 mm = 108 mm` 到镜前基准 | 需以探测平面成像为准 |
| 13 | `relay lens -> camera` | `10-11` | `254.0-279.4 mm` | `260 mm` | 与配置一致 |

说明：

- 上表中“当前配置值”优先以 `physics_cfg.yaml` 为准。
- 若平台照片、手工孔距计数与配置存在轻微差异，应优先保证成像关系和探测平面定义正确。
- `4f` 的总有效长度应按成像公式理解，而不是只按机械底板尺量理解。

## 6. 不可随意压缩的物理约束

### 6.1 `4f relay system`

- `4f` 段是成像定律决定的刚性结构。
- 若随意缩短，会直接破坏傅里叶平面关系，导致像差、模糊和分辨率下降。
- 在代码侧，它对应 `macro_optics.f_4f` 以及与 `4f` 两侧相关的传播距离。

### 6.2 `SLM` 照明段

- 扩束后的光束需要稳定、均匀地填充 `SLM` 有效区域。
- 这部分约束被 `OpticalPathVAE.encode()` 保留为理想前级光路语义。

### 6.3 芯片后 `3 mm` 探测平面

- 探测平面定义仍然是关键 retained 结论之一。
- 代码中对应 `macro_optics.dist_chip_to_plane = 0.003`。
- relay lens 的作用是把这个探测平面准确成像到相机。
- 在新主线中，该平面作为中间监测量存在，不作为主重建损失平面。

## 7. 对新主线仍然有效的结论

迁移到 `OpticalPathVAE + latent diffusion` 后，本文仍然保留以下价值：

- 作为物理平台和历史实验系统的背景说明
- 作为 `physics_cfg.yaml` 字段含义的解释文档
- 作为第一阶段 `encode()` 和 `decode()` 光路顺序的基线资料

不再由本文承担的内容：

- 第一阶段概率瓶颈说明
- 扩散训练主链路说明
- 损失函数说明
- checkpoint 选择策略说明

## 8. 一句话结论

`guanglu.md` 在新主线下继续 retained，但角色收敛为“光路与物理参数基线文档”；第一阶段 `OpticalPathVAE` 会直接引用其顺序、距离和读出平面语义。
