# V6 全 3000 张完整重训合同

## 已确认决策

Full 主线不再从 V6 Fold0 e34 续训。正式模型只从经过 SHA-256 审计的公开
Co-DINO ViT-L checkpoint 初始化，在全部 3,000 张官方标注图上完整训练 V6
全模型。这里的“从头”指 V6 训练 lineage 从公开权重开始，不是随机初始化，也
不是只训练检测头。

旧的 `co_spec_dino_vitl_full.py` 与 `train_v6_full.sh` 原样保留，作为 e34
低学习率 continuation 的可复核备份；它不再是 Full 主线，也不得误启动为本次
实验。

## 初始化与数据

- 唯一外部 checkpoint：公开 Co-DINO ViT-L O365-to-COCO；SHA-256
  `733d2ccde180a55151a68a6cab7c9f42b117d24d38d6197b37caf3189243256c`。
- V4、V5、V5R、V6 e34、Soup 和 SWA 权重均不得进入本次 lineage。
- AdamW、EMA 和 FP16 scaler 均 fresh；普通 resume 必须完整恢复三者。
- 训练集为 `full.json` 中全部 3,000 张官方图；无 200 张 sentinel、无本地独立
  validation，也不以训练 loss 冒充泛化指标。
- Full 的两个 class-balanced loss buffer 在 update 0 前由 3,000 张标注的真实
  18 类计数确定。它们是固定 buffer，不产生新的 optimizer 状态。

## 预算、阶段与 checkpoint

global batch 为 2，因此每轮是 1,500 attempted updates。Fold0 的训练阶段按
完整数据曝光等比例乘 `3000/2400 = 1.25`：

| 坐标 | Full 配置 |
| --- | --- |
| LR warmup | `0/.02 -> 1875/1.0` |
| stabilize | `0–7,500` exposure updates |
| capacity + 16-band Mosaic | `7,500–70,500` exposure updates |
| clean | `70,500–100,500` exposure updates |
| AP75/弱类 polish | `100,500–120,000` exposure updates |
| LR tail | `70,500/.30 -> 100,500/.06 -> 115,000/.015 -> 120,000/.004` |

LR、ViT/融合打开进度和 EMA 只按成功 AdamW step 前进；AMP overflow 不算成功
更新。数据增强只能在 dataloader epoch 边界安全切换，因此按已完成的全数据曝光
轮次切换，避免一次 AMP skip 把整个增强阶段错误地推迟一轮。

默认软上限是 80 epoch / 120,000 attempted updates。无 early stopping。每 2 轮
保存一次完整 raw+EMA+AdamW+scaler checkpoint，且不自动删除。因此 e34、e40、
e46、e52、e60 和后续 checkpoint 全部保留；预计 40 份 checkpoint 约占
`230–240 GiB`，当前 `/data` 空间足够。

Fold0 的 600 张验证曲线已经用于训练时长选择。预先锁定：

- e34 / 51,000 attempted updates：主候选；
- e40 / 60,000、e46 / 69,000、e52 / 78,000、e60 / 90,000：备选。

Full 训练 loss、梯度和 EMA 距离只诊断训练健康，不能改写上述模型选择。完整训练
到 e80 不会覆盖早期候选。由于所有标注图都参与训练，任何 Full checkpoint 都不
得被表述为“本地验证最佳”。

## 启动与续训

正式训练只由用户手动启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
V6_FULL_SCRATCH_CONFIRMED=YES \
V6_FULL_SCRATCH_MAX_EPOCHS=80 \
bash tools/launch_v6_full_scratch_detached.sh
```

命令使用 `nohup + setsid + stdin=/dev/null`，退出 SSH 后继续运行，但停电或主机
重启仍会终止。再次执行同一命令时，只允许从 Full scratch 工作目录的
`latest.pth` 完整续训；Fold0/e34/旧 continuation checkpoint 会被门禁拒绝。

若 e80 后有足够外部证据需要延长，设置更大的
`V6_FULL_SCRATCH_MAX_EPOCHS` 并运行同一命令。120k 以后保持最低 LR 和 polish
阶段，禁止另起一条重头训练。

## 耗时、选择和推理

Fold0 的实测训练中位数约为 `1.905 s/update`。120,000 次 attempted updates 的
纯训练估算约 63.5 小时；加 checkpoint、首次构建和 I/O 后预计约 64–68 小时。
正式首轮完成后应用实际吞吐重新估计，不把该时间当停止条件。

训练结束后先对预锁候选做完整 checkpoint/EMA/哈希审计，再使用冻结冠军协议
（1280 identity + horizontal、Gaussian Soft-NMS、global max300）横向比较和
生成 CSV。不得在 Full checkpoint 选择阶段混入 Soup、full TTA、切片、WBF 或
逐类校准。
