# HW207 实物光路与固定 CMOS 2×2 ROI

本目录定义本会话的独立校准工作包。用户确认当前光路和器件已装好并接通；这个状态单独记录为 `ASSEMBLED_USER_CONFIRMED`。器件铭牌、光传播端口、当前参考面距离、Relay 映射、分类平面、CMOS像素注册和读出参数仍须由可复查数据确认。

## 固定分类几何

- CMOS 目标窗口：200×200 µm，坐标原点在窗口左上角；X 向右、Y 向下。
- ROI-0：X/Y 为 35–85 µm；ROI-1：X 为 115–165 µm、Y 为 35–85 µm。
- ROI-2：X 为 35–85 µm、Y 为 115–165 µm；ROI-3：X/Y 为 115–165 µm。
- 每个 ROI 为 50×50 µm；横纵间距30 µm；四边各35 µm。
- 读出只使用四个ROI的像元面积重叠积分与 `argmax`；平局或总能量为零时弃权。像元到物理窗口的仿射变换必须显式输入，不能默认相机居中或 1:1。

## 当前验收状态

当前为 `DIAGNOSTIC_ONLY / PREFLIGHT_BLOCKED`。按6.5 µm参考像元间距换算，固定在CMOS感光面上的每个50 µm ROI约宽7.692像元，200 µm窗口约宽30.769像元；这些是规格换算，不是窗口位置、读出模式或PSF的实测结果，ROI像素宽度也不随光学倍率改变。光学倍率改变的是分类面光斑在CMOS上的投影尺寸与位置。历史C3.5/C3.7/C3.8使用约669.5 µm边长的大ROI，因此旧准确率不能回答本固定50 µm ROI任务。

主模型必须等 P0–P4 的实测接口、相机标定和独立数据角色闭合后建立。此包提供固定ROI面积积分、几何计算、静态DMD采集图案和采集说明，不生成CMOS亮斑、不运行历史forward、不重训相位、不控制硬件。

## 可复现入口

使用项目支持的 Windows Python 环境，从仓库根目录运行：

```powershell
& 'D:\ANACONDA\envs\optical-generative-models-win\python.exe' scripts/research/lightgen/hw207_physical_cmos_v1.py `
  --config configs/research/lightgen/hw207_physical_cmos_v1.json `
  --output-dir outputs/20261106_hw207_physical_cmos_v1/runs/diagnostic-only_01a1a9b1-5c54-7d81-9999-0cb3b09557a2/artifacts_attempt_02
```

运行根目录预先保存了完整的 `git_status_before.txt`；`artifacts_attempt_02/` 子目录必须不存在，入口拒绝覆盖已有结果。输出包括历史距离表与当前测量模板、理想 Relay/f30 情景敏感性、相位候选文件元数据核查、固定 ROI 软件接口、拓扑/几何/敏感性图，以及静态 DMD 采集图案。生成结果保持 `DIAGNOSTIC_ONLY / PREFLIGHT_BLOCKED`：没有真实 CMOS 强度图、冻结相位 forward、训练结果或硬件采集。图案为原生 2560×1600 的 8-bit 灰度 PNG（仅0/255），需要实验人员依照硬件安全规程手动采集。
