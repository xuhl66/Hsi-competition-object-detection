# HOD26 Agent Working Rules

## 唯一目标

在完全遵守竞赛规则、最终保持单模型提交的前提下，所有技术决策都以最大化
Kaggle 私榜 `mAP@[.50:.95]` 为唯一目标。AP50、速度、参数量和实验形式只作
诊断，不得凌驾于最终分数。

## 训练纪律（最高优先级）

- 战术上激进，工程上严谨。允许结构级大改、长时间双卡训练和所有合规提分
  手段；不得用保守预算替代充分训练。
- “训练时间宝贵”是要求一次把高潜力模型训练到可信收敛，避免欠训练、碎片化
  Pilot 和反复从头重跑，不是要求缩短必要训练。
- Smoke test 只验证数据、代码、显存、DDP、反向、EMA、checkpoint 和推理链路。
  短训不得用于判定模型上限，也不得把未收敛模型端给用户作架构结论。
- 训练预算必须先换算为 optimizer updates。不得照搬其他数据集的 epoch 数，也
  不得用任意的 36/60/96 轮阈值草率淘汰架构。
- 高潜力架构必须使用一个从开始就设计完整的长程训练：足够高的软上限、连续
  学习率与增强阶段、EMA、定期验证、完整 checkpoint，并能无缝延长而不重启。
- 只有当主指标在多个验证窗口内形成可信平台、学习率已到计划低位、最佳点不在
  训练边界，且 AP75、小目标和弱类趋势也不再上升时，才可称为收敛。若最佳点
  仍在最后一次评估或曲线仍上升，必须视为未收敛。
- 软上限不是停止命令。到达计划尾部仍在提升时，应从同一 checkpoint 连续扩展
  低学习率阶段，禁止因为最初 horizon 不够而从头重练。
- 只有数据/标签/坐标错误、非有限损失、不可恢复的训练故障或已经满足上述收敛
  证据，才允许提前终止正式长训。
- 架构结论应比较相近 optimizer updates / 样本曝光，并最终比较各自充分收敛的
  最佳 checkpoint。重大版本可合并所有有强证据的高收益改动，不为科研归因性
  把长训拆成多个保守小实验。

## 当前必须牢记的教训

V2 的 36 轮仅有 3,924 次 optimizer updates，且第 36 轮仍刷新最佳
`mAP@[.50:.95] = 0.69607`。它是一次完成的工程链路检查和早期训练记录，不是
V2 上限，也不能据此判定 V2 失败。原“36 轮达到 0.760 才晋级”的门槛作废。

V3 曾用 `--tuning` 从 e228 只加载模型权重，丢弃 AdamW 的 `step`、
`exp_avg`、`exp_avg_sq` 以及既有 EMA 训练年龄；失忆的 optimizer 随即带着
强增强破坏已经形成的优化方向。2026-07-30，V5 首次启动又因把“跨架构初始化”
误当成普通 fine-tune，险些从 V4 e30 只继承权重而重置 optimizer/EMA。该进程
已停止并废弃。这是一次流程级失误，今后不得依赖用户再次提醒。

2026-07-31 原 V5 正式训练在 e18 达到 `bbox_mAP=0.719` 后发生定位支路
退化。根因不是网络中断：未监督的 coarse box 被当作固定 FDR 锚点；共享
decoder 的 FDR 运行时引用污染了 Co-DETR auxiliary forward；FDR/LQE 被错误
置于 8× LR；update0 的原生 V4 与 update1 的自定义坐标函数也不连续。原 V5
e18 及其全部 checkpoint 只能作故障证据或推理诊断，永久禁止进入后续训练
lineage。修复版必须使用独立且有 matching+DN 损失的 pre-bbox head、原生 Aux
隔离、连续的 prediction/reference 坐标、定位 1× LR，以及不平滑 transition
控制量的 post-successful-step EMA。

### 跨版本状态继承硬门槛

- `vN -> vN+1` 默认是能力连续升级，不得默认视为 fresh fine-tune。正式长训前
  必须逐个可训练张量分成：同名同坐标、可证明的代数重参数化、语义父分支克隆、
  真正新增/确实不兼容四类。
- 对前三类，在数学上有效时必须同时迁移 raw 权重、EMA 权重及年龄、AdamW
  `step/exp_avg/exp_avg_sq`；只写“从父权重初始化”不算完成状态继承。新版本
  可以使用从零开始的本地 LR/增强/结构过渡坐标，但这绝不等于允许旧参数的
  optimizer step 归零。
- “语义父分支克隆”不自动等于 moments 可迁移。克隆权重/EMA 后，只有梯度坐标
  和尺度也有充分依据时才迁移 AdamW moments；否则该分支必须列为“权重继承、
  optimizer fresh”。当前 V5 的 15 个多尺度 salience moment clone 是已经开跑
  lineage 的显式启发式选择，不得被后续版本无条件照抄。
- 只有逐项列名并统计数量、参数量占比的真正新增或无法合理映射的张量可以 fresh。
  如果重大结构变化确实使旧 moments 不可用，必须在正式训练前报告丢弃范围、
  理由、风险和替代保护，并取得用户明确确认；禁止静默重置。
- transition、warmup、低学习率和暂缓强增强只能降低结构迁移冲击，不能替代
  optimizer/EMA 状态继承，也不能作为重置旧状态的理由。
- 跨版本 smoke 必须分两段：先从父代派生的完整 stateful checkpoint 做真实
  optimizer updates 并保存，再从该子代 checkpoint 恢复继续。检查必须证明：
  旧参数 `step = 父代 step + 成功更新数`、新参数 `step = 成功更新数`，EMA
  raw/影子权重与继承年龄均存在，第二段 epoch/iter/optimizer/EMA 连续。仅有
  forward/backward、有限 loss 或普通子代续训成功，不足以放行正式长训。
- 每次正式训练前的初始化报告必须明确写出父 checkpoint 及 SHA-256、父
  epoch/iter、继承与 fresh 的 optimizer 张量数和参数量占比、EMA 年龄、
  FP16 scaler 是否继承，以及上述两段式 smoke 的断言结果。
- 必须区分 runner attempted iterations 与真正成功的 optimizer steps。AMP
  overflow 跳过的 step 不能算成功更新；调度坐标若仍按 runner.iter 前进，必须
  同时报告 skip 数。V4 e30 是 `iter=42180`、AdamW `step=42168`，并有 12 次
  AMP skip，不得再把两个数字混写成同一个“optimizer updates”。
- FP16 scaler 必须作为独立状态检查。父 checkpoint 有 scaler 时，继承或重置都
  必须显式声明父值、策略和理由；跨图 fresh scaler 不得再写成“父代未保存”。
  bootstrap 可以按已声明策略重新校准，子代普通 resume 则必须恢复 scaler。
- 正式加载前必须 fail-closed：严格核对模型与 EMA 的完整 name/shape/dtype、
  optimizer group/name/order/shape/超参数和 finite moments、step 族、lineage、
  scaler 及 checkpoint 哈希。不得依赖 MMCV 的 `strict=False`，也不得依赖
  PyTorch 按参数组位置静默 zip 的默认行为。
- epoch 边界 checkpoint 不保存 Python/NumPy/CUDA/worker RNG 和 sampler 游标时，
  只能称为训练状态连续，禁止称为 bitwise resume。卡数、参数组超参数或源码
  schema 变化必须由门禁拒绝，不能假设普通 resume 会采用新配置。
- 当前 V5 lineage 为兼容父代，EMA 在 optimizer.step 前更新且平滑 transition
  buffer。它会显著压低早期验证中的 transition，只能作为已声明的当前 lineage
  行为继续恢复；EMA 时序或 control-buffer 策略的改造必须放到新版本边界，不能
  在同一训练中途暗改。

## 冠军基线与版本节奏（冻结）

- 当前公榜冠军控制组固定为 V4 Co-DINO ViT-L 的 epoch-30 EMA 单 checkpoint：
  全图宽度 1280 的 identity + horizontal 两视图，同类 Gaussian Soft-NMS
  (`iou_threshold=0.55`, `sigma=0.5`, `pre_score=output_score=0.0001`)，最后按
  每图全类别置信度取 `max_per_image=300`。其 Fold0
  `mAP@[.50:.95]=0.71711748`，公榜为 `0.67686`。
- 这是一条已经由公榜确认的完整推理基线，不是裸模型。V5 及后续
  V6/V7/V8/... 在模型能力持续提升阶段，必须让各自的单一收敛 EMA checkpoint
  使用同一推理协议；这里的“同一”指相同输入视图、Soft-NMS 和 top-k，不是沿用
  V4 的权重。
- 模型迭代阶段不得把 Soup/SWA、多尺度或 12/16-view full TTA、垂直/双向翻转、
  切片、WBF、box voting、逐类阈值或测试分布校准混入版本晋级结果。单视图可作
  模型诊断，正式横向比较与提交一律使用上述冻结协议。
- 0730_b 将 e30 换为 e54 并同时改成 12 views、`sigma=0.7`、`max500`，本地
  `+0.00620` 但公榜 `-0.00087`；0730_c 在 e30 上四切片本地仅
  `+0.000385`、公榜 `-0.00004`。因此不得用同一验证集上的复杂后处理最优值
  推翻已由公榜确认的控制组，也不得一次提交同时更换 checkpoint 与推理协议。
- 当前战略是连续推进 V5/V6/V7/V8/... 的模型级能力，`0.70+` 公榜只是阶段
  里程碑，不是收益上限或达到即停的目标。只有模型已按训练纪律可信收敛、连续
  重大版本也不再产生可靠提升时，才暂时冻结模型更新并观察排行榜。
- 模型端形成可信平台后，才重新研究常规推理优化；常规手段也穷尽后，才允许在
  用户逐项明确授权、竞赛规则专项复核、单模型与最终审查均可复现的前提下，重新
  评估总纲中冻结的合规边界手段。作弊、漏洞利用、标签重建和绕过限制永久禁止。

## 操作边界

- 训练由用户手动启动；Codex 负责设计命令、检查日志、诊断和改进，不擅自启动
  长训或长期监控。
- 未经用户明确要求，不上传 Kaggle CSV。
- 所有项目改动只在本仓库目录及其已挂载的 `storage` 路径内完成。
- 每次正式训练前必须报告：初始化权重、总 optimizer updates、调度阶段、验证与
  checkpoint 频率、可续训设计、预计耗时及停止条件。

## V6 Full 全数据决策（冻结）

- Full 主线从经审计的公开 Co-DINO ViT-L 权重开始，在全部 3,000 张
  官方标注图上完整训练 V6 全模型；不继承 V6 Fold0 e34，不使用 200 张
  sentinel 选 checkpoint。旧 e34 低学习率 continuation 仅作备份。
- global batch 2 时为 1,500 attempted updates/epoch；默认 80 epoch / 120,000
  attempted updates 软上限。LR、融合和 EMA 按 successful updates 前进，数据增强
  按已完成的全数据曝光轮次切换。
- 预先锁定 e34/e40/e46/e52/e60 为候选，其中 e34 为主候选；完整训到 e80
  但不把最后一轮默认为最佳。Full 没有独立验证集，训练 loss 不得用于宣称
  泛化最优。
