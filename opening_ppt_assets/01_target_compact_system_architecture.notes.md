# 目标紧凑 SMF–OLS 非线性光学语义生成架构

## 设计说明

1. 横向3:1原生SVG，透明画布，适合16:9白底PPT中央放置。
2. 四个衍射薄相位面、100通道SMF阵列、端面薄膜与CMOS使用不同器件形状。
3. 橙色仅强调F_NL及其说明，薄膜贴合SMF输出端面；没有材料名称或响应曲线。
4. 灰色虚线表示拟定光路，浅蓝光束不表达实测光强或模场分布。
5. 每个目标衍射层100×100格点、2 μm节距，活动区约200 μm×200 μm。
6. 阵列正面为10行×10列共100芯，侧面线条为纵向结构示意，不是独立激活单元。
7. 中文Noto Sans SC、英文Arial、F_NL为Cambria Math；SVG保留可编辑text/tspan。
8. 输入3/5/8与输出3为手绘矢量图标，不是MNIST或模型结果；输出标“示意”。
9. 器件大小、fiber芯径/间距、膜厚、距离和光束宽度非按物理比例；没有比例尺。

## TARGET / CURRENT evidence supported

**TARGET：** 整张拟制造紧凑光路，2L Encoder+2L Decoder作为目标结构，双光子打印衍射层，端面拟集成物理F_NL，“输入/语义条件”与清晰数字输出的目标功能。器件形状与数字不是实物或性能验收证据。未添加角度输入、材料身份、膜厚、绝对功率或性能值。

**CURRENT evidence supported：** 当前基础研究代码已有2+2相位栈、100×100相位格点、2 μm采样、10×10=100维复OLS投影/传输/重发及CMOS强度读出的仿真定义。200 μm×200 μm由100×2 μm推导。可选材料模型代码不证明真实器件已测量或语义收益。已有主要研究端点使用2+4混合窗口，不能由本TARGET图替代。

## 科学一致性审计

同名qa.json消费最终SVG DOM、渲染文字、端口/相位面计数、器件顺序、贴合坐标、PNG尺寸/alpha、80%PPT宽度的字号。交付须同时通过自动检查和完整图/PPT预览目视复核。

旧skill validator固定要求SLM、Te、1.5 mm文字和较窄aspect ratio，与用户此次要求不兼容。本图不为通过旧模板而添加无要求的器件/材料/参数；其固定拓扑检查标为不适用，用户指定拓扑和九项科学边界由本图渲染检查实际消费。skill文件未修改。

## 证据来源

- [统一规范](D:/codex-original/opening_ppt_assets/VISUAL_STYLE_GUIDE.md)。
- [开题审计](D:/codex-original/OPENING_PPT_EVIDENCE_AUDIT.md)：TARGET与当前大模型边界。
- [基础运行时](D:/codex-original/src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py)：10×10端口、100×100格点、2 μm、Encoder/Generator各2层常量。
- [SMF实现](D:/codex-original/src/ldm_project/v2/optics/fiber_operator.py)：100复overlap系数、transport与重发。
- [参考配置](D:/codex-original/configs/reference/lightgen_power_proxy.yaml)：参考/未标定参数，不作为实测值。
- [正式v2合同](D:/codex-original/docs/architecture/v2_runtime.md)：基础2+2/100×100/2 μm；identity OLS不是SMF验收。
- [缩放与打印边界](D:/codex-original/docs/archive/v67_ideal_phase_boundary.md)：设计参数与制造标定分开。
- [物理前向记录](D:/codex-original/docs/physical_power_integration_v1.md)：可选模型已实现、诊断未通过，无实物验收。

## 复现

作者脚本build_01_target_compact_system_architecture.mjs。CLI接收已配置Node模块目录与Chrome/Chromium可执行文件；不将机器路径固化为项目运行时参数。依赖playwright、sharp。只构造/渲染本图，不读取数据、运行模型、访问网络或训练。
