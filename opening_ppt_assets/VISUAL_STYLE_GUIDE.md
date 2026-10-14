# 开题 PPT 科研插图统一视觉规范

版本：v1；建立日期：2026-10-13。消费者：本项目后续开题PPT概念图、光路图、器件局部图、代理模型说明图及其导出/检查流程。本轮只建立规范，不生成任何SVG、PNG或第一张图，不修改运行时、配置、历史报告或已有skill。

## 1. 风格依据和优先级

1. 用户本次明确要求和当前科学证据优先。
2. 已配置的 `nature-figure` skill：采用证据先于美化、克制配色、清晰主次、可编辑文字、最终尺寸检查的规则。已读取manifest、core contract/stance、design theory、Nature版式观察及QA规则。
3. 已配置的 `optical-nature-schematic` skill：采用原生SVG、可编辑 `<text>`、明确光学器件形状、真实拓扑、准确10×10端口、端面附着材料和科学状态区分。已读取project figure contract与validator的拓扑要求。

可访问仓库工作树中未检出内置 `SKILL.md`；上述为当前环境已配置、带项目光学合同的既有技能，不称其为仓库内新建skill。沿用现有风格，不另外创造商业信息图体系。

本轮是规范初始化，不选择Python/R数据绘图后端，不调用ImageGen/OpenRouter，不生成AI位图。后续概念图由原生矢量构造；若另有定量绘图任务，按nature-figure的明确/已保存后端规则执行。

## 2. 科学状态：必须先于配色和绘制确定

每张图在制作记录中写明：图所回答的问题、状态、所消费的配置/代码/结果、允许显示的数值、缺失参数和可接受结论。图内只放短标签；完整来源和说明放配套记录或PPT备注。

| 状态 | 定义 | 图中约定 |
| --- | --- | --- |
| CURRENT | 有当前源码支持的实现，或有明确保存结果支持的研究端点 | 正常结构实线；同时说明是实现、仿真还是结果，不自动等同于实物验收 |
| TARGET | 本次开题拟实现的目标结构/功能 | 未完成部件或连接用灰色虚线，必要短标签“目标架构”“拟集成”“待标定” |
| LEGACY | 历史实验、旧假设和算法代理 | 单独放置/分组，短标签“历史代理”；不得混入当前硬件主链 |

CURRENT不要求整图为实线：例如已实现仿真光路中的未标定实物映射，必须局部灰色虚线并标“待标定”。颜色不是状态的唯一依据，结合线型与短标签；不能用鲜艳颜色暗示已经验证。

### 本次读取的证据及边界

- [开题审计](D:/codex-original/OPENING_PPT_EVIDENCE_AUDIT.md)：2026-10-12证据快照；本轮将其与2026-10-13源码及文档核对，不能将旧快照当成永久最新事实。
- [当前状态](D:/codex-original/docs/CURRENT_STATUS.md)：研究/诊断最新停止点，不从局部数值效果推断实物完成。
- [LightGen基础运行时](D:/codex-original/src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py)：基础2+2、100×100、2μm、532nm；默认LP01；新增Gaussian选择与默认关闭的physical_optics配置。
- [已有大Generator](D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_generator_large_capacity_quick_comparison_v1.py:148)：已有主要研究端点使用2层Encoder+4层混合窗口Generator；Decoder窗按光路为100²/400²/100²/400²。只有声明绘制这一CURRENT端点时才使用该结构。
- [SMF/OLS实现](D:/codex-original/src/ldm_project/v2/optics/fiber_operator.py)：复数面积加权overlap、100复系数、T和模式重发；旧局部端口与新全局参考几何必须分别绑定配置，不混用尺寸。
- [历史Strict ReLU](D:/codex-original/scripts/research/lightgen/execute_lightgen_2d_ideal_facet_relu_long_v1.py:323)：重发复场的像素级阈值强度代理。标LEGACY，不画成Te测量，更不能画Te=ReLU。
- [新材料响应代码](D:/codex-original/src/ldm_project/v2/application/research/lightgen/material_nonlinearity.py)：CURRENT可选工程实现，包含identity、SA、SA+RSA、振幅加相位模型；方程参数不是实测Te/CoSi/TaP性能。仅在消费明确代理场景的图中显示；不自动放入已有训练端点。
- [物理功率集成记录](D:/codex-original/docs/physical_power_integration_v1.md)：小2+2可配置研究前向已实现，诊断有效未通过，G3/G9未通过，未训练；SA扫描是探索性代理证据，不能升级为材料验收。
- [参考物理配置](D:/codex-original/configs/reference/lightgen_power_proxy.yaml)：249.3μW、T=0.889、125μm pitch等带reference/unmeasured状态。不能画成项目实物测量；本轮不在视觉规范中固化这些参考数值。
- [正式v2合同](D:/codex-original/docs/architecture/v2_runtime.md)及[配置](D:/codex-original/configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml)：identity OLS无光纤，是另一个正式兼容路径，不能替代本课题SMF研究线。
- [旧2+2打印边界文档](D:/codex-original/docs/archive/v67_ideal_phase_boundary.md)：项目自定义缩放设计和历史理想相位加工候选，LEGACY参考；不证明现有打印实物。
- [LightGen本地补充材料](D:/codex-original/research_assets/lightgen/science_adv7434/paper/sciencesm.pdf)：已核对PDF第2–3、7页。论文2+4、400²/600²、3μm及Gaussian公式属于论文参考，不替换项目参数。仓库 `research_assets` 中未检出正文PDF，不声称已阅读全文或以补充材料替代正文证据。

### 开题目标结构

本次TARGET为2层Encoder+2层Decoder、每层约100×100 optical neurons、2μm pixel、10×10 SMF/OLS及拟集成端面非线性。可以画为目标结构，但必须与已有CURRENT混合大模型分开。100×100是相位格点，10×10是fiber端口；2μm不是fiber pitch、已验证打印线宽或光纤芯径。

尺寸、间距、波长、膜厚、MFD、材料阈值、功率和响应曲线只有在所绘对象有明确证据时才标。缺失参数写“待标定”或省略，不填经验数。未经标定的Gaussian示意必须标“模场示意”，不暗示已测半径。

## 3. 固定配色

| 角色 | 色值 | 用法 |
| --- | --- | --- |
| PPT背景 | `#FFFFFF` | 白底展示；素材画布本身透明 |
| 主深蓝 | `#203F72` | 当前主光路、主要器件轮廓、关键短标签 |
| 次级灰蓝 | `#7186A5` | 次级结构、辅助连接、弱化但已有的部件 |
| 浅灰蓝填充 | `#EDF2F7` | SLM/相位面/CMOS等主体的平面填充 |
| 普通灰 | `#6B7280` | 辅助文字、TARGET虚线、未完成映射和待标定项 |
| 材料/非线性唯一强调色 | `#D98945` | F_NL、候选材料薄层、明确非线性代理；不能用于普通模块或泛指不确定性 |
| 风险红 | 少量、仅风险语义 | 未固定新红色色号，不额外扩充结构调色板；只用于风险/警示，不能作为普通结构色 |

状态与强调色叠加时：未完成的材料模块可有橙色薄层/填充，但其未验证连接仍用灰色虚线；不把TARGET主路径整体变成橙色实线。橙色表示材料/非线性，不表示性能更好。

背景、器件和卡片不做渐变；只允许光束、Gaussian模场使用柔和、局部渐变，不用发光边、强光晕或粒子。没有阴影和商业卡片排版。不使用黑底、霓虹色、赛博朋克、厚黑框、卡通/拟人AI图标或无物理含义芯片纹理。

## 4. 字体与文字

- 中文遵循光学skill的Noto无衬线方向：优先`Noto Sans CJK SC`；当前Windows已发现并读名字表确认的同系列简体中文字体为`Noto Sans SC`。实际本机渲染用`Noto Sans SC`，后备`Source Han Sans SC`、`Microsoft YaHei`。不将不存在的字体写成已成功使用。
- 英文、数字、缩写：`Arial`，后备`Helvetica`和无衬线。保持SMF、OLS、CMOS、F_NL大小写，不混用多种装饰字体。
- 数学公式：`Cambria Math`或可用的STIX数学字体，符号采用规范serif/math样式；变量斜体、单位正体。尽可能以SVG `<text>/<tspan>`保持编辑性；复杂公式确需文字转路径时，先明确字体兼容原因，不默认转路径。
- 图内只用中文短标签，必要时使用SMF、OLS、CMOS、F_NL。不放长段解释、完整论文引用、完整run路径或大标题。
- 以实际PPT放置尺寸检查字号，不沿用论文的7–9pt小字号。主要标签以18–22pt为起点，辅助标签14–16pt；优先删减文字，不靠过度缩小字号塞入。数学上下标也须在投影尺寸可读。
- 来源、公式解释、参数身份和科学边界放配套记录/PPT备注；需要区分状态时，图内保留最短必要标签。

## 5. 矢量结构与箭头

- 概念图使用原生SVG，主体为平面矢量。沿用skill的optical-components primitives；所需symbol内嵌到文件，禁止依赖外部SVG的fragile use引用。
- SLM、衍射层、SMF、端面薄膜、传播空间、CMOS绘制为不同物理对象，不统一画成同样的软件方框。衍射层是薄平面，允许稀疏神经元格点；不使用电子电路纹理。
- 正面完整SMF阵列必须为准确10×10=100芯。局部剖面可减少示意fiber数量，但须短标签“局部放大”；没有实测几何则不按假定芯径/pitch画比例尺。
- 如果明确绘制拟定材料端面，材料膜贴合SMF输出端面，不漂浮成独立软件盒。视觉放大厚度须在备注说明“非按比例”；未知膜厚不标数值。
- 箭头统一使用小实心箭头头部。主路径深蓝，明确的非线性路径橙色，TARGET/未来/待标定路径灰色虚线；局部信息箭头不能遮挡器件边界和文字。
- 主箭头最终PPT尺寸线宽约2–2.5pt，统一以2.25pt为起点；器件轮廓较细、辅助引线更细，不用厚黑边。按实际放置缩放后复查，不将SVG逻辑px当成最终PPT pt。
- 图以光学连续性、留白和清晰主次组织；不强行使用固定三卡/四卡商业布局。多panel若用于比较，保留可比尺寸、对齐和一致间距。
- root记录 `data-figure-mode` 与 `data-scientific-status`，重要groups有实际语义的 `data-stage`；不得为让旧validator通过，把非Te材料强行命名为`te-film`。

## 6. 导出与文件约定

输出根目录：`D:/codex-original/opening_ppt_assets/`。

每张后续图建议同名保存：

```text
<figure_id>.svg        可编辑源文件
<figure_id>.png        透明背景预览，宽度至少2000px
<figure_id>.qa.json    实际执行的检查与发现
<figure_id>.notes.md   CURRENT/TARGET/LEGACY、证据来源与参数身份
```

- SVG保留 `<text>`，不把整图栅格化嵌入，不默认把文字转不可编辑path；字体后备和渲染器共同保证中文可读。
- SVG画布透明，无全幅白色background rectangle；素材默认放在白色PPT页面上。这同时满足白底视觉与透明PNG。
- PNG保持原图aspect ratio，宽≥2000px，有有效alpha通道；透明边缘不能出现黑底或白底烘焙。
- 另以白底合成视图进行视觉QA，作为检查时的临时预览，不取代透明PNG，也不冒充新图。本轮不生成任何预览。
- 不更改原证据图以制造性能差异；引用已有结果时保留同口径数据与源角色。

## 7. 每张图必须执行的QA

本节是后续交付要求，初始化阶段没有图，不能声称已有渲染或QA通过。

1. 先检查状态和证据：拓扑、层数、端口数、参数归属，是否将TARGET/LEGACY/参考参数误画为CURRENT实测。
2. 保存SVG后必须实际自动渲染，再检查SVG解析、viewBox、文字编辑性、内嵌资源、PNG尺寸及alpha。
3. 按最终字体测量文字边界与画布/分组空间；检查溢出、标签重叠、箭头穿字或遮挡端面。数学上下标一起检查。
4. 检查渲染的中文缺字/方框、字体fallback、笔画和数学符号；文件可解析不能代替字体渲染检查。
5. 检查完整阵列100芯、传播方向、材料贴合端面、必要比例解释、器件大小层级、留白和整体比例。
6. 预览同时检查至少2000px原图及实际PPT显示尺寸；自动检查不能替代最终渲染的目视核对。
7. 写入实际QA结果；任一阻断项未解决不得标passed。修订后重新渲染和重查，不消费旧版本的绿色结果。

`optical-nature-schematic/scripts/validate_svg.py`具有固定2+2及Te相关拓扑/文字要求。只能对其合同适用的图执行并解读；CURRENT 2+4或“材料待定”图不能盲套其模板，也不能把模板token检查当成科学真实性证明。对应图的自动几何/文字/拓扑检查必须消费该图明确声明的对象；不可审计的项目须如实记录，不伪称通过。本轮不修改该skill或validator。

后续定量Python/R多panel若纳入PPT，另按nature-figure运行最终rendered panel alignment及PDF文字/碰撞检查，保留数据来源、人口/seed角色与不确定性定义。对无图的风格规范不执行这些绘图检查。

## 8. 已解决差异与待确认项

已按用户明确要求解决，无需重复确认：

- skill原始色值与本PPT色值不同：使用本规范#203F72/#7186A5/#EDF2F7/#6B7280/#D98945，保持skill的克制视觉语法。
- skill原则不使用渐变：用户明确允许光束/Gaussian柔和渐变，仅这两类例外。
- 白底与透明PNG：透明素材置于白色PPT，QA用白底合成视图。
- skill固定2+2与CURRENT 2+4：2+2用于明确声明的目标/基础模型；CURRENT端点按其真实配置绘制，不静默覆盖。
- 旧审计“没有材料接口”与新源码：新增SA等可选工程接口属于当前代码实现；没有实测材料性能、未通过的物理诊断不能升级为真实器件实验。
- 论文3μm、2+4与项目目标2μm、2+2：各属自己的证据对象，不混写。

唯一尚需用户选择的命名：是否将TARGET端面材料统一写为“Te（候选）”，还是“F_NL（材料待定）”？在用户未明确固定材料前，默认采用后者；不画第一张图，不自动排序Te/CoSi/TaP性能。

未标定膜厚、fiber几何、功率、材料传输和相位不属于可凭美学决定的参数；后续图缺证据时省略或写“待标定”，而非要求初始化阶段补造数据。
