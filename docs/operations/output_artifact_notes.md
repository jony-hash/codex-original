# D2NN 主线输出与可视化说明

本文件只解释生成图片、tensor bundle 和光学诊断。v1 Stage 命令见 [Stage Guide](stage_guide.md)，v2 current 命令见 [v2 runtime](../architecture/v2_runtime.md)，完整文档导航见 [Documentation](../README.md)。

## 恢复历史 outputs

当前 Windows 工作区的 `outputs/` 同时包含两类内容：当前 Windows-native
运行结果，以及从不可变 WSL tar 归档恢复的历史可见副本。恢复副本中的
checkpoint、图像、JSON evidence、日志和诊断文件仍保持原始历史身份；它们不
自动进入当前 runtime、formal acceptance 或 checkpoint resolution。

恢复完整性以 `outputs/_wsl_restore_metadata_v4/` 下的
`restore-plan.json`、`restore-state.json` 和 `path-map.jsonl` 为准。文件数量、
磁盘占用或目录修改时间只是描述性指标，不能替代恢复验证。原始 tar、index
和 manifest 位于外部不可变归档位置，不得由当前工作区改写。

`outputs/version_index.json` 与 `outputs/version_index.md` 是恢复的历史 WSL
索引，可能包含 `/home/...` 绝对路径；它们只能用于 provenance，当前 Windows
工具必须使用显式 Windows 路径或当前 run 元数据，不能自动把这些路径解析为
当前工作区文件。

## v2 Conditional Artifacts

v2 current run 在 `outputs/YYYYMMDD_v2/runs/<run>/` 下保存 evaluator、identity
initialization、conditional evidence、raw-CMOS review tensor 和 candidate dataset shards。
candidate manifest 必须保持 `status: candidate`；只有 generation admission、memorization
和 utility 全部通过后才写入 `accepted_manifest.json`。这些 artifact 不得写入
`outputs/final_pt/`，也不能与 v1 teacher/distillation bundle 混用。

## v2 Research Artifacts

Research lifecycle outputs also use persistent versioned `outputs/` run paths,
but remain explicitly `synthetic_diagnostic_only` or equivalent. A prepared run
manifest records intended inputs and actions; it is not validation evidence.
Only artifacts whose producer status and declared consumer both pass may reach
the next registered research action. Research checkpoints, closeouts and
presentation bundles never enter `outputs/final_pt/` and cannot warm-start the
formal identity chain unless a new formal contract explicitly authorizes them.

## v1 Teacher 与蒸馏数据

Teacher first-stage / diffusion 只用于训练和导出蒸馏目标。新 run 默认写入真实版本目录 `outputs/YYYYMMDD_<version>/runs/<run_name>/`；其中日期前缀取该版本当前已知最早 run 日期。相关图像仍可能出现在：

- `outputs/YYYYMMDD_<version>/runs/<run_name>/images/first_stage/`
- `outputs/YYYYMMDD_<version>/runs/<run_name>/images/diffusion/`
- `outputs/YYYYMMDD_<version>/runs/<run_name>/images/samples/`

这些图用于检查 teacher 是否能提供可用的 `camera` target，不代表最终 D2NN 硬件推理路径。

## v1 D2NN One-shot 输出

主线严格复光场输出位于：

- `images/*_optical_field_amplitude.png`
- `images/*_optical_field_phase.png`
- `artifacts/*_optical_fields.pt`
- `artifacts/*_optical_field_provenance.json`

这些图只从同一传播复场 `E = real + i·imag` 在渲染时计算：振幅为原始 `abs(E)`（共享线性色标、仿真 `a.u.`），相位为原始 `angle(E)`（固定 `−π…π` 弧度）。相位 PNG 会遮罩相对峰值强度低于 `−60 dB` 的像素；未遮罩的原始相位仍保留在 tensor artifact。它们是数字孪生中的传播复场，不是硬件实测场，也不表示绝对 SI 场强。

SLM command、generator desired intensity、target code、mask transmittance 和 CMOS intensity 都不是传播复场，必须以独立 control / device-mask / CMOS diagnostic 命名，不能放进上述光场图。`relay_psf` 的 camera 为 intensity-only：严格复场链在 chip 截止，绝不输出或暗示 camera phase。

若 target-code、bridge 注入或 `relay_intensity_psf` 破坏复场连续性，provenance 会标记 `field-chain-invalid`，完整链图不会写出；最多只允许 intervention 前、实际存在的复场平面。v48–v50 旧的完整 physical-chain amplitude/phase 图已由 v51 追溯报告标为 `visualization-harness-invalid`，不作为光场证据。

`camera/student` 是训练和评估的主输出。由 v55 建立并由当前正式合同保留的几何固定 SLM/CMOS 为 `200 × 200 @ 2.5 µm`（500 µm FOV），formal encoder/decoder metasurface 为 `100 × 100 @ 2 µm`（200 µm FOV）。SLM→Encoder 使用倍率 `−0.4` 的相干 4f relay，完整 SLM FOV 映射到完整 Encoder 面；不再使用自由传播后的中央 aperture 截取。LightGen 仅指隔离的 reference/diagnostic proxy，不是 formal encoder/decoder 的运行时名称。不要把 raw D2NN camera diagnostics 和 loss 图像尺寸混为同一个物理平面，也不要把 raw diagnostics 当成另一条产品输出路径。

## Raw Tensor Bundle

`*_optical_fields.pt` 只保存严格光场链的原始 `real/imag`、shape 和 provenance；renderer 由这些 tensor 现场计算 `abs(E)` 与 `angle(E)`。`one_shot_sample_tensors.pt`、`one_shot_distill_eval_tensors.pt` 和训练 artifact 仍可保存 control、mask、intensity 与其他 raw diagnostics，但它们不是严格复光场证据。

v1 写入责任已收敛到 `ldm_project.v1.student.artifacts`（预览与 device diagnostics）和 `ldm_project.v1.student.evaluation`（strict field provenance）；v1 `application/` 只提供 run manager/输出布局，不改变 artifact 字段语义。v2 current/research 使用自己的 evidence 和 lifecycle writers，不能与这些 v1 tensor bundle 名称互相推断兼容性。

- `camera_plane`：用于 loss 和主评估的 `200 × 200` 相机强度输出，是 one-shot student 的主输出。
- D2NN raw diagnostics：内部衍射层与 high-res camera 的原始诊断 tensor；其尺寸由 `one_shot.d2nn.grid_size` 和 chip grid 决定。
- device-mask diagnostics：mask transmittance 的 `|t|` 和 `arg(t)`，描述器件参数而非传播光场。

`dense_linear` 仍是保留 baseline，但主线应使用 `one_shot.optical_core_arch: diffractive_phase_only`。判断 D2NN 是否正常时，优先看 `reference/teacher` 与 `camera/student` 是否一致，再看 SLM、encoder、OLS、decoder 和 chip 的振幅/相位诊断定位问题。
