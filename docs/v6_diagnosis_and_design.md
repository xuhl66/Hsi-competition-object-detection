# V5 全量诊断与 V6 独立方案

状态日期：2026-08-09

工作分支：`agent/v6-independent`

结论状态：V5/V5R 已淘汰；V6 已实现并通过最终源码哈希绑定的双卡 exact smoke、
Fold0 audit-only 和 all-3000 continuation audit-only。正式长训尚未启动，等待用户
手动执行 detached 启动命令。

## 1. 结论先行

V5R 不是因为训练轮数不足而失败。它在训练集上的匹配 IoU 和回归损失持续改善，
但验证集 AP75、弱类 AP 和 Kaggle 公榜同时下降；这是已经形成证据闭环的泛化失败。
继续修 FDR、LQE 或 salience 只是在一个失效方向上增加复杂度，不能成为 V6。

V6 定为一个独立的单模型 `CoSpec-DINO ViT-L`：

1. 只以公开的 Co-DINO ViT-L Objects365→COCO checkpoint 为外部初始化；
   不加载 V4、V5 或 V5R checkpoint，optimizer 和 EMA 从 V6 自己的 update 0
   开始。
2. 保留已经证明上限很高的 Co-DINO ViT-L 主检测器，彻底替换 V4/V5 的弱光谱
   旁路、标量门控、FDR/LQE 和无效 salience。
3. 新建波长感知的 HSI 主干、高分辨率双流融合及带完整分类/定位监督的光谱提议
   路由，让光谱特征真正影响 encoder proposal 和 decoder query，而不是只产生一个
   很小的 residual。
4. 用 Align-DETR 的质量对齐分类思想解决 AP75，不再移植 D-FINE 的 FDR。
5. 用物理合理的传感器域增强、弱类等权正样本目标、鲁棒性验证门禁和 V6 内部的
   全 3,000 图阶段解决跨分布泛化。

这不是 V5 的修补版，也不继承 V5 的故障训练血缘。

## 2. 可复现证据入口

只读诊断命令：

```bash
storage/envs/codino-v4-smoke/bin/python tools/diagnose_v6_evidence.py \
  --initial-checkpoint \
    storage/pretrained/v5r/v4_e30_stateful_to_v5r_init.pth \
  --trained-checkpoint \
    storage/v5r/protected_candidates/v5r_fold0_epoch40_macro_0.717700.pth \
  > /tmp/v6_evidence.json
```

脚本只读取 COCO 视图、训练日志和 checkpoint，不改数据或权重。它输出数据尺度、
各类样本数、V4/V5/V5R 指标、训练窗口、参数变化量、光谱融合门以及公榜对照。

## 3. V5/V5R 全量诊断

### 3.1 总指标没有形成真实提升

| 模型 | Fold0 mAP | AP50 | AP75 | APs | 弱四类均值 | 公榜 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 e30 EMA | 0.71711748 | 0.969 | 0.845 | 0.681 | 0.500475 | **0.67686** |
| 故障 V5 e18 | 日志 0.719 | 0.970 | 0.853 | 0.684 | 0.504900 | 0.67112（非冻结协议） |
| V5R e40 EMA | 0.717700 | 0.969 | 0.849 | 0.681 | 0.489325 | **0.66977** |
| V5R e80 EMA | 日志 0.711 | 0.967 | 0.836 | 0.676 | 0.471650 | 未提交 |

V5R e40 对 V4 的本地 macro 只有 `+0.00058252`，相同冻结推理协议下公榜却
`-0.00709`。因此不能归咎于 TTA、Soft-NMS 或 CSV；变化来自模型本身及其
泛化能力。

故障 V5 e18 的定位支路存在已确认的坐标、共享 decoder 和监督错误，永久禁止
进入任何训练 lineage。它只保留作故障证据。

### 3.2 本地微增完全来自强类，比赛真正的弱点反而退化

V4→V5R e40：强 14 类均值 `+0.008443`，弱四类均值 `-0.011150`，强弱差距
从 `0.274032` 扩大到 `0.293625`。

| 类别 | V4 e30 AP | V5R e40 AP | 变化 |
| --- | ---: | ---: | ---: |
| car | 0.5900 | 0.5628 | **-0.0272** |
| e-bike | 0.5619 | 0.5447 | **-0.0172** |
| people | 0.4623 | 0.4449 | **-0.0174** |
| stone_block | 0.3877 | 0.4049 | +0.0172 |
| apple_plastic | 0.7788 | 0.8002 | +0.0214 |
| egg_wood | 0.7467 | 0.7725 | +0.0258 |
| orange | 0.7517 | 0.7810 | +0.0293 |

也就是说，V5R 的整体数值被容易类抬高；这正是原晋级判据的盲区。后续晋级不能
只看 macro AP，必须同时审查 AP75、APs、弱四类和逐类收益来源。

### 3.3 FDR 已经退化为近乎恒等映射

训练窗口均值显示：

| epoch | main matched IoU | pre matched IoU | FDR refinement L1 |
| ---: | ---: | ---: | ---: |
| 2 | 0.8778 | 0.8799 | 0.01248 |
| 10 | 0.8904 | 0.8911 | 0.00055 |
| 40 | 0.9046 | 0.9045 | 0.00068 |
| 80 | 0.9267 | 0.9261 | 0.00045 |

从 e10 开始，FDR 对 box 的平均修正只有约 `4e-4~7e-4`；主框和 pre-box 的
matched IoU 几乎相同。checkpoint 中 FDR 原始参数相对 L2 变化只有 `2.55%`，
LQE 只有 `1.24%`。这不是有潜力但没训够，而是该支路在本任务上选择了“不动”。

### 3.4 salience 学了很多参数，但没学到强前景判别

V5R e40 相对初始化：多尺度 salience 参数相对 L2 变化 `48.77%`，说明它确实
被充分更新；但 e40 的平均 foreground/background response 约为
`0.1112/0.1040`，分离度很弱。继续增加 salience 层或调 loss 权重没有高收益
证据。

### 3.5 光谱支路没有死亡，但对主检测器影响太弱

V5R e40 的光谱 encoder/fusion/proxy 相对 L2 变化分别约 `18.32%/15.65%/22.60%`；
所以“光谱完全没训练”不是根因。真正的问题是其注入主干的有效
`tanh(gate)` 绝对均值从 P2 到 P5 只有：

```text
0.07258, 0.05404, 0.03732, 0.03309
```

而且越到语义层越小。最终决策仍主要由三通道代理和 ViT-L 决定，光谱旁路缺少
完整类别/框监督，门控自然可以通过接近零来规避风险。

### 3.6 V5 对传感器 band 顺序的假设有误

本项目的 X2Cube 解码按 4×4 马赛克 row-major 产生 band 0–15。V5 认为这些只是
“位置 ID”，不能作为波长顺序，所以把 V4 的相邻差分从 31 通道证据改成 raw16。

XIMEA `xiSpec2 TechnicalManual V2.00` 第 4.1.2 节在同一 VIS3 传感器条目中给出
row-major index 0–15，并依次给出中心波长：

```text
464.5, 472.8, 480.2, 489.3, 499.0, 508.2, 516.3, 526.1,
534.7, 544.3, 552.3, 561.8, 571.2, 580.5, 588.1, 597.2 nm
```

因此 V6 应把 band id 固定到物理波长，用非均匀间隔的 `dI/dλ` 和 spectral token
位置编码；不能继续假设无序，也不能只恢复一个未经尺度校正的简单差分。

参考：[XIMEA xiSpec2 Technical Manual](https://www.ximea.com/getattachment/9ca53218-192e-4701-bafa-3cf17375c82e/xiSpec_TechnicalManual-DWL_manual.pdf)、
[imec VIS3 传感器规格](https://www.imechyperspectral.com/en/offering/standard-spectral-chips/snapshot-ssm4x4-vis-spectral-sensor-chip)。

### 3.7 e40→e80 是明确过拟合，不是低学习率定位阶段

e40→e80：

- 训练 matched IoU：`0.9046 → 0.9267`；
- 训练 `loss_bbox`：`0.0767 → 0.0552`；
- 验证 mAP：`0.718 → 0.711`；
- 验证 AP75：`0.849 → 0.836`；
- 验证弱四类：`0.4893 → 0.4717`。

模型对训练分布拟合得更紧，却对验证分布更差。V6 必须把跨分布鲁棒性放进表示、
增强和晋级门禁；单纯延长当前 V5R 不会解决。

## 4. 本地限制下的主要瓶颈（按重要性）

### P0：验证与晋级目标失真

fold0 的类实例比例很平衡，但弱类的独立场景极少：

| 类别 | train 独立图 / box | val 独立图 / box |
| --- | ---: | ---: |
| car | 153 / 452 | 40 / 113 |
| e-bike | 144 / 369 | 31 / 92 |
| people | 285 / 876 | 67 / 219 |
| stone_block | **34 / 218** | **8 / 54** |

把 stone 图片重复 2×/4× 只增加同一场景曝光，不增加泛化信息。当前验证对
`stone_block` 的方差尤其大，macro 的千分位差不能可靠预测公榜。

### P1：光谱物理信息没有成为检测决策主路径

band 顺序假设错误、光谱门控幅度小、只有弱 mask/salience 监督，导致最有机会
区分材料和跨照明稳定的 16 波段没有被充分利用。

### P2：小目标和高 IoU 是剩余主分数区间

fold0 train 的 8,056 个框中：

- COCO small：5,655，`70.1961%`；
- medium：2,306，`28.6246%`；
- large：95，`1.1792%`。

V4 的 AP50 已达 `0.969`，而 AP75 为 `0.845`。继续提高“有没有检测到”收益小，
主要空间来自小框定位、分类/IoU 置信度对齐和弱类精排。

### P3：场景多样性与官方数据利用不足

V4/V5/V5R 都只训练 fold0 的 2,400 张，最终冠军模型没有使用另外 600 张有标签
官方图。对只有 34 个训练场景的 stone_block，这 20% 数据不能永久闲置。

### P4：训练分布太容易被记住

现有 repeat sampler 加强了弱类图的重复，但没有生成新场景；单个 fold 的总体 AP
又允许强类收益掩盖弱类损失。V6 需要传感器域扰动、强弱类分解门禁和最终全数据
阶段共同解决，而不是提高 repeat factor。

## 5. V6 最终架构：CoSpec-DINO ViT-L

### 5.1 唯一外部初始化

只允许以下一个公开 checkpoint：

```text
URL:
https://huggingface.co/zongzhuofan/co-detr-vit-large-coco/resolve/main/pytorch_model.pth
SHA-256:
733d2ccde180a55151a68a6cab7c9f42b117d24d38d6197b37caf3189243256c
```

它是官方 Co-DETR/Co-DINO ViT-L Objects365→COCO 模型。官方模型表报告 COCO val
65.9 AP、test-dev 66.0 AP，高于同表 O365→COCO Swin-L 的 64.1；因此 V6 不为
“换主干”而降级到 Swin-L。参考：[Co-DETR 官方仓库与模型表](https://github.com/Sense-X/Co-DETR)。

加载策略：

- 映射 ViT-L、SFP、encoder/decoder、proposal 和可语义对应的类别行；
- 所有 V6 HSI 模块从随机初始化开始；
- V6 的 visual backbone、detector、HSI/fusion 和 heads 全部参与训练；这里的
  “从公开权重从头训练 V6”绝不是冻结 backbone、只训练 detection head；
- optimizer、EMA、AMP scaler 全部 fresh；
- 这是公开预训练模型到新任务的正常 fine-tune，不是 V4→V6 resume，不适用
  “必须继承 V4 optimizer moments”的跨版本门槛；
- 初始化 manifest 仍保存 URL、SHA-256、上游 commit、映射明细和未加载张量清单。

禁止加载以下任何 checkpoint：V4 e30、故障 V5 e18、V5R 任意 epoch、任何 soup
或 CSV 反推产物。

### 5.2 波长感知 HSI 编码器（SpecDETR 核心思想）

实际 `HOD26V6WavelengthViT` 的光谱流同时编码四种互补表示：

1. 全训练集 robust band normalization 后的 raw16；
2. 每像素去强度/L2 归一化的 spectral-shape16，降低曝光和照明强度漂移；
3. 按真实 `Δλ` 计算的 15 维 `dI/dλ`，保留材料光谱边缘。
4. 由 4 组正弦/余弦构成的 8 维固定波长 basis，显式注入非均匀物理波长位置。

光谱空间部分采用 SpecDETR 启发的 bounded self-excitation residual block，在
高分辨率特征上放大细小目标响应；最终代码没有声称实现其原论文完整 subpixel
模块。为了在双 4090 上可行，光谱 encoder 在主图 1/2 空间尺度运行，并输出
P2/P3/P4；主 ViT-L 仍保持 1280 宽度。光谱编码器保持 FP32 数值岛，新卷积块采用
batch-size-one-safe FP32 GroupNorm，ViT-L、融合和检测头继续使用 AMP。

SpecDETR 是 2025 ISPRS P&RS 的 HSI 专用 detector，官方代码和权重已公开；其
SPOD 报告为 0.856 mAP、0.863 AP75，模型约 16.1M 参数。V6 使用其成熟的光谱-
空间编码思想，但不加载 SPOD 权重，避免第二个外部数据血缘和 30→16 band
checkpoint 适配风险。参考：[SpecDETR 官方仓库](https://github.com/ZhaoxuLi123/SpecDETR)、
[论文](https://arxiv.org/abs/2405.10148)。

### 5.3 真正的双流融合，不再允许光谱门自动归零

旧 `base + tanh(gate) * spectral_delta` 删除。新融合遵循 S2ADet 已验证的
spectral-spatial 双流聚合思想，在 P2/P3/P4 使用显存可承受的双向 DCNv2
可变形采样融合：

- visual query 从 HSI 特征取样材料/边缘证据；
- HSI query 从 visual 特征获得形状和语义；
- 融合 residual 在前 6k successful updates 由确定性系数从 0 平滑到 1，之后
  固定为 1，不能通过学习一个接近 0 的标量永久逃避光谱；
- control coefficient 不进入 EMA 平滑。

S2ADet 的 HOD3K 数据为 3,242 张、512×256、16 band、470–620 nm，与本赛的
3,000 张、约 502×245、16 band、460–600 nm 高度接近。其官方实现没有发布
S2ADet 训练权重，因此只采用论文中的双流结构证据，不引入额外 checkpoint。
参考：[S2ADet 官方仓库](https://github.com/hexiao0275/S2ADet)、
[论文](https://arxiv.org/abs/2306.08370)。

### 5.4 光谱提议路由：直接监督弱类和小目标

在融合后的 P2/P3/P4 上建立光谱特征，并连续下采样形成 P5/P6/P7；训练期
one-to-many 光谱检测路由具有完整的 18 类分类、centerness 和 box loss，不再只
监督一个无类别 mask。该路由的高质量正样本坐标按 Co-DETR collaborative
assignment 注入主 decoder query。

效果目标：

- 小目标在进入主 decoder 前已有高分辨率、类别相关的位置种子；
- 光谱支路若不工作会直接增加分类/box loss，不能靠缩小 fusion gate 隐身；
- 推理时只保留一个共享主 decoder 的最终输出，训练辅助路由不形成模型集成。

DQ-DETR 的 categorical counting/dynamic query 数量模块不整套加入。它解决的是
AI-TOD 中一张图 tiny object 数量高度不均衡的问题，而本数据每张标注目标中位数
只有 3、P90 只有 6，固定 1,500 queries 的“数量不足”并非瓶颈。V6 只解决 query
位置和特征质量，不为不匹配的问题增加计数模块。参考：[DQ-DETR（ECCV 2024）](https://github.com/hoiliu-0801/DQ-DETR)。

### 5.5 AP75：采用 Align Loss，删除 FDR/LQE

主 decoder 保留 Co-DINO 已成熟的迭代 L1+GIoU box refinement；分类正样本改为
IoU-aware Align Loss，使分类置信度显式反映框质量，解决高分错框在 AP75 下排序
靠前的问题。Co-DETR 已有 one-to-many auxiliary assignments，所以不重复移植
Align-DETR 的整套 matching，只采用与本地问题直接匹配的质量对齐 loss。

Align-DETR 在 BMVC 2024 公开代码中报告 R50 12ep 50.3 AP / 54.8 AP75，24ep
51.4 / 55.8；论文针对的正是 classification-regression 与跨层 target
misalignment。参考：[Align-DETR 官方代码](https://github.com/FelixCaae/AlignDETR)、
[BMVC 论文页](https://bmvc2024.org/proceedings/211/)。

V6 不包含：FDR、GO-LSD、LQE、coarse-box anchor、V5 salience、V5 transition，
也不继承其任何参数或 optimizer state。

### 5.6 弱类与 macro AP 目标

- 删除弱类图片统一 repeat2；标准 sampler 保持独立场景均匀洗牌。
- matched-positive 分类和光谱辅助路由按 effective-number 做 class-balanced
  weighting，使用平方根和上限，避免 stone_block 被无界放大。
- 背景、DN negative 和 box loss 不做粗暴按类复制。
- 每次验证固定输出 18 类 AP、弱四类均值、strong14 均值、APs/AP75；任何整体
  增益若只来自 strong14，不允许晋级。

### 5.7 跨分布泛化

只用官方训练图进行以下物理一致增强：

- 所有 band 同步空间变换；
- 全局曝光、真实波长上的平滑照明斜率；
- 有界 per-band gain/offset、shot/read noise、轻度 blur；
- spectral-shape 分支提供强度不变表示；
- capacity 阶段保留 V4 已验证的 native 16-band Mosaic，clean/polish 阶段关闭；
- MixUp 继续关闭，避免小框边界和材料光谱被线性污染。

验证除原始 fold0 外，额外报告固定的 4 种传感器扰动视图 robustness mAP；它只作
泛化门禁，不替代官方原始 mAP，也不参与手工逐类阈值拟合。

## 6. 已固化的训练设计

### 6.1 Fold0 能力与收敛阶段

标准 2,400 图、global batch 2 时约 1,200 attempted updates/epoch。以 update 为
唯一调度坐标，当前软上限设计为 96,000 attempted updates：

| successful update 区间 | 目的 |
| --- | --- |
| 0–6k | 公开 detector 稳定、HSI/fusion 打开；无 Mosaic |
| 6k–56.4k | 主容量学习；多尺度、Mosaic 0.25、传感器域增强 |
| 56.4k–80.4k | 1152/1280 clean refinement；关闭 Mosaic |
| 80.4k–96k | 1280 低学习率 Align/AP75/弱类 polish |

设计依据：V4 在 42,168 successful updates 左右达到 e30 最佳；V6 新增从零开始且
有直接监督的 HSI 主干和辅助路由，给出超过两倍的可信学习窗口。96k 是软上限，
不是停止命令；若边界仍刷新 AP75、APs、弱类或 macro，必须从同一 checkpoint
延长低学习率尾段。

- detector/decoder peak LR：`1.25e-5`（1×）；
- visual ViT-L 顶层 peak LR：约 `1.25e-5`，24 层按 `0.90` LLRD，最早层约
  `9e-7`；这不是冻结主干；
- 新 HSI/fusion/aux route peak LR：`1.0e-4`（8×）；
- 重新映射的类别坐标 peak LR：`5.0e-5`（4×）；
- LR 由 successful AdamW update 驱动：`0.02→1.0` warmup 到 1.5k，6k 前平台，
  再按 `0.30@56.4k / 0.06@80.4k / 0.015@92k / 0.004@96k` 连续余弦下降；
- 使用 FP64 汇总的稳定全局 L2 gradient clipping (`max_norm=0.1`)，避免 3.68 亿
  参数的 FP32 norm reduction 溢出后静默清零梯度；
- EMA：只在成功 optimizer step 后更新，control buffer 不平滑；
- validation/checkpoint：每 2,400 attempted updates；
- checkpoint 包含 model/raw+EMA、optimizer、scaler、attempted/successful/skip
  计数及源码/配置哈希；
- 普通 resume 必须通过 V6 fail-closed gate 和两段式 smoke。

80 epoch 是 96,000 attempted update 的软上限；若 AMP skip 使 successful update
尚未走完调度，或边界指标仍上升，必须从同一 checkpoint 增加 epoch，不得重头。

### 6.2 全 3,000 官方图阶段

2026-08-13 经用户确认，Full 主线改为从唯一公开 Co-DINO checkpoint
初始化，在全部 3,000 张官方标注图上完整重训 V6 全模型。V6 Fold0
e34、V4/V5/V5R、Soup/SWA 均不进入该 lineage。旧 e34 低学习率
continuation 实现原样保留为备份，不再是 Full 主线。

Full 以 1,500 attempted updates/epoch 运行 80 epoch / 120,000 attempted
updates 软上限。Fold0 的学习率、融合、增强和 EMA 时间尺度按每张图曝光
等比放大 `3000/2400=1.25`。预先锁定 e34/e40/e46/e52/e60，并训练到
e80 保留完整窗口。不保留 200 张 sentinel：它只能发现崩溃，无法可靠选择
18 类 macro mAP checkpoint。详细门禁和启动命令见
`docs/v6_full_scratch_contract.md`。

### 6.3 最终工程放行证据

- 唯一公开源 checkpoint SHA-256：`733d2ccd...256c`；
- 派生 V6 public init SHA-256：`65f606fd...3357`；provenance SHA-256：
  `b37bb377...5915`；
- 最终 runtime source-lock SHA-256：`143ad356...f481`，锁定 24 个运行文件；
- 最终 exact smoke：`storage/v6/smoke/exact_20260809_041209`；1280×640、双卡、
  Mosaic phase 1 → checkpoint → clean phase 2 resume；
- phase 1/2 为 `4/4 → 8/8` successful AdamW updates，AMP skip 为 0，EMA age
  为 8，scaler 为 1.0，1026 个 optimizer state 的 step/moments 全有限且连续；
- 单卡峰值显存 `21,124 MiB`；GroupNorm 修复后前四步 pre-clip global norm 为
  `1361 / 1108 / 636 / 720`，不再由零填充区域的 LayerNorm bias 支配；
- 26 项 V6 单元/诊断测试、Python compile、8 个 shell 的 `bash -n` 均通过；
- 预计 Fold0 96k soft horizon 约需 `55–75` 小时（含 40 次验证与 checkpoint）；
  正式首个完整 epoch 后应按实测稳态吞吐重算，不把该估计当停止条件。

## 7. 晋级与提交门禁

V6 的单一 EMA checkpoint 先用冻结冠军协议：1280 identity+horizontal、同类
Gaussian Soft-NMS (`iou=0.55`, `sigma=0.5`)、threshold `0.0001`、max300。
不在晋级阶段重新调 TTA/NMS。

晋级必须同时满足：

1. Fold0 macro mAP 对 V4 的差异超过固定重复运行/评估噪声；
2. AP75、APs、弱四类均值不下降，并且弱类不是整体收益的负贡献源；
3. 原始验证和传感器扰动 robustness 都无明显退化；
4. 最佳点不在软上限边界，或已从同一 checkpoint 延长到可信平台；
5. 同一冻结协议的 CSV 才允许提交 Kaggle。

不把 `+0.03` 或 `+0.05` 写成收益上限。V6 是针对四个已量化根因的架构级升级；
期望的是百分位级而不是千分位级跃迁，但任何具体分数只能由充分收敛后的同协议
结果证明。

## 8. 独立提交契约

最终 V6 交付包只包含：

```text
v6/
  README.md
  requirements.lock
  configs/
  src/
  upstream/              # 固定 commit 或可校验安装脚本
  tools/prepare.py
  tools/train.py
  tools/infer.py
  tools/check_submission.py
  provenance/public_checkpoint.json
  weights/FINAL_V6.pth
```

硬约束：

- `src/hod26/v6`、`configs/v6`、V6 tools 不得 import `hod26.v4/v5/v5r`；
- 可以复制并整理通用 X2Cube/COCO/metric 代码到 V6 namespace，但最终包不能依赖
  旧版本目录或旧 checkpoint；
- 公开初始化下载 URL、SHA-256、许可证和参数映射必须可复核；
- 最终只加载一个 V6 EMA checkpoint，训练辅助路由不产生第二套推理权重；
- README 从官方原始数据到 CSV 的命令必须在干净环境 smoke；
- 保存最终参数量、FLOPs、源码 commit、权重/CSV SHA-256 和 Phase 2 两分片同权重
  证明。

Git 分支保留历史不影响提交独立性；审查方拿到的是上述导出的干净 V6 包，而不是
整个实验仓库的所有历史版本。

## 9. 仍需持续追踪（当前无需再次确认）

1. **外部预训练规则风险（最高）**：公开可下载不等于主办方已经允许外部数据
   预训练。此前用户已授权在主办方未回复时继续这条提分路线，所以 V6 按公开
   Co-DINO 实施；但进入最终获奖审查前仍应取得书面许可。若明确禁止，必须切换
   同架构的 official-data-only 初始化，V4 也会面临同一资格风险。
2. **最终全数据血缘已更新**：采用“唯一公开 Co-DINO → 全 3,000 图完整
   V6 重训”。不得从 Fold0 e34、smoke、V4/V5/V5R 或任何 Soup/SWA
   checkpoint 进入 Full scratch；普通 resume 则必须完整恢复该 Full lineage 的
   model/EMA/AdamW/scaler。
3. **SpecDETR 权重继续禁用**：只采用其公开架构思想，不加载 SPOD checkpoint，
   保持唯一外部权重来源；不再等待用户确认。
4. **显存门禁已通过**：最终 1280×640 双卡 exact smoke 峰值为
   `21,124 MiB/GPU`，保留了 ViT-L、P2、半尺度 HSI encoder 与 activation
   checkpointing。正式训练不得静默增大 batch/canvas 或缩减结构；若环境变化
   导致 OOM，必须先报告再调整。

## 10. 计划与实际实现对照

核心计划均已实现：16-band 物理波长编码、P2/P3/P4 双流融合、P2–P7 直接监督
光谱 proposal、Align-DETR 风格 IA-BCE、弱类 effective-number 权重、物理传感器
增强、capacity-only native HSI Mosaic、successful-update LR、post-successful-step
EMA、鲁棒性门禁、冻结冠军推理协议和 stateful all-3000 continuation。

最终实现相对早期文字方案有四项有意修正：

1. “deformable cross-attention”落实为双向 DCNv2 sampling fusion，以便在双 4090
   上保留 1280 输入和 P2；不是删减为标量 gate。
2. SpecDETR 采用 self-excitation 思想但未伪称实现完整 subpixel 模块；波长位置由
   真实 `dI/dλ + 8D basis` 表达。
3. 早期草案的 visual 0.1× / new 4× 被实际配置修正为 ViT 0.90 LLRD、new 8×、
   class 4×；上述最终值已由真实反向 smoke 验证数值健康。
4. exact smoke 额外发现并修复 IA-BCE FP16、光谱 Jacobian FP16、逐像素 LN
   零填充梯度爆炸、MMCV FP32 global-norm 溢出、EMA total-iter 和 checkpoint
   metadata dispatch 六类工程缺陷；这些不是可忽略告警。

明确未加入且不是遗漏：MixUp、重复弱类图片、OSSDet 二值目标 mask（已由更强的
18 类+box 直接光谱监督替代）、DQ-DETR counting、FDR/LQE、Soup/SWA/full TTA。
后处理仍冻结为 V4 公榜验证过的两视图 Soft-NMS 协议，先隔离验证模型级收益。

## 11. 研究选择记录

- 采用 Co-DINO ViT-L：官方公开通用检测上限最高且已在本地夺冠。
- 采用 SpecDETR 表示思想：与 HSI、小目标、AP75 直接匹配，2025 期刊、代码公开。
- 采用 S2ADet 双流证据：其 16-band HOD3K 与本地传感器/尺寸最接近。
- 采用 Align Loss：直接解决 AP75 的分类/定位排序失配，模块化且成熟。
- 不采用完整 DQ-DETR counting：本地目标数很少，query 数量不是瓶颈。
- 不采用 TinyFormer 作为 V6 主干：2026 新预印本和公开权重有潜力，但仓库/复现
  仍过新，且其 O365→COCO XL 62.5 AP 低于现有 Co-DINO ViT-L 65.9；可作为后续
  V7 候选，不值得让 V6 放弃已验证主干。
- 不采用 Mr.DETR 主干：其 multi-route 思路与 Co-DETR 现有 collaborative
  assignment 高度重叠，公开 Swin-L 结果没有形成足够大的替换证据。

## 12. 2026-08-11 e34 推理与类别校准结论

V6 e34 EMA 的原生单视图提交 `0811_b.csv` 得到公榜 `0.67685`，与冻结的 V4
冠军 `0.67686` 基本持平，但没有刷新冠军。随后仅针对 V6 排序失配诊断类别校准：

- 预 top-k 的逐类 margin 虽在 Fold0 有小幅收益，却使测试预测中 `car` 占比升至
  80.2%；该版本 `0811_c.csv` 在上传前即被分布门禁拒绝，永久不得提交。
- 最终实验固定 e34 原生的框、类别和每类数量，只把已保留检测的分数乘以
  `(p_class / (p_class + max_other)) ** alpha_class`。逐类参数要求 Fold0 的三个
  image-id 分折全部提升；`stone_block` 虽有 54 个框但只来自 8 张图，因此冻结。
- 稳健筛选后的 Fold0 mAP 为 `0.72308327`，相对原生 `0.72212284` 提升
  `0.00096043`；三个分折分别提升 `0.00113542 / 0.00065930 / 0.00053687`，
  AP75 从 `0.84625470` 到 `0.84773951`。框、标签和类别计数均保持不变。
- 对应提交 `0811_d.csv` 的公榜只有 `0.67445`，相对原生 `0811_b` 下降
  `0.00240`。完整重推还观察到低置信 Soft-NMS 尾部的 CUDA 非确定性
  （`0811_b` 为 299,999 行，本次为 300,000 行），因此不能把全部下降精确分解
  到校准；但公榜已经明确否决该组合，不能进入冠军协议。

冻结结论：当前公榜冠军仍是 V4 e30 的 `0.67686`。V6 以及后续模型晋级继续使用
既定控制协议，不再使用这套 query-margin 类别校准。单一 Fold0 内再分三份只能
降低明显过拟合风险，不能替代真正独立的 OOF/跨域校准证据；后续不得用同类
Fold0 小收益消耗提交额度。
