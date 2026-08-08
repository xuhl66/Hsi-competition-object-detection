# HOD26 最终提交与代码审查准备

状态日期：2026-07-27。这里记录的是版本无关的交付流程，不把 V3 或任何中间
候选误标为最终模型。V4、V5 或后续版本继续追加候选记录；最终只交付实际胜出
的一个模型 checkpoint 及其对应代码。

## 官方交付要求

当前[竞赛规则](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/rules)
要求：

- Phase 2（2026-09-23 至 2026-09-25）提交一个 CSV，同时覆盖原 test 1000 张
  和届时释放的 ranking 1000 张；两部分等权，最终排名完全由私榜决定。
- 前六名须在 2026-09-27 前向 `hottracking2025@gmail.com` 提交完整代码包：
  训练与推理源码、训练权重、可复现提交 CSV 的推理脚本、环境/依赖/命令
  README，以及模型参数量和 FLOPs。
- 最终预测必须来自单模型。当前流水线用一个 checkpoint 的一套权重分别推理
  test/ranking，不做多 checkpoint 融合或 WBF。

## 两个必须保留书面证据的官方歧义

1. [Evaluation 页面](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/overview/evaluation)
   当前声明八列：
   `id,image_id,class_id,confidence,x1,y1,x2,y2`；但 2026-07-27 重新下载的
   官方 `sample_submission.csv` 仍只有后七列。现阶段继续生成从 0 连续编号的
   `id`，因为带 `id` 的 CSV 已被 Kaggle 接收并计分。Phase 2 数据发布后必须
   重新下载 sample、复核页面，并保存组织者对列格式的书面答复。
2. 当前规则对外部数据有限制，但组织者尚未书面澄清公开 Objects365/COCO
   预训练权重。任何由该权重派生的最终模型都必须完整声明来源、URL、commit
   和 SHA-256；最终提交前必须取得并保存组织者的书面许可。没有书面许可时，
   不能把“讨论区无人反对”视为允许。

## 为什么必须记录 e228

如果最终胜出模型由 e228 继续训练，e228 是初始化父 checkpoint，必须写入训练
血缘；e228 又源于公开的 Objects365→COCO 权重，因此该外部祖先也必须披露。
这不构成集成：最终推理仍只加载一个后代 checkpoint。血缘记录的目的，是让
审查者解释并复现权重来源，而不是把每个中间候选都提交为最终模型。

已启动的 V3 候选记录位于：

```text
storage/v3/compliance/hf_hsi_deim_x_fold0/
  launch_manifest_20260727T160052+0800.json
```

它明确标记为 `candidate_training_stage`，不是 final。

## 后续每次正式长训前

候选版本可以继续增长，不要求每个候选都 commit 或 push。为防止未提交源码在
后续修改中丢失，先运行通用快照命令；命令只写审计记录和源码快照，不启动训练：

```bash
storage/envs/deim-v2/bin/python -m hod26.audit \
  --candidate vNEXT_candidate_name \
  --command '完整的手动训练命令' \
  --config configs/vNEXT/train.yaml \
  --source src/hod26/v2/train.py \
  --source src/hod26/v2/extensions.py \
  --source src/hod26/v2/compat.py \
  --source tools/train_vNEXT.sh \
  --data storage/v2/dataset/fold0/train.json \
  --data storage/v2/dataset/fold0/val.json \
  --data storage/cache/band_stats.json \
  --parent-checkpoint storage/path/to/one_parent_checkpoint.pth \
  --lineage-manifest storage/path/to/previous_candidate_manifest.json \
  --external-pretraining-provenance \
    storage/pretrained/v2/deim_dfine_x_object365.pth.provenance.json \
  --initialization 'parent raw+EMA+AdamW state mapped by name/reparameterization; only enumerated genuinely new tensors fresh; explicitly state FP16 scaler handling'
```

输出默认写入 `storage/compliance/candidates/<candidate>/`，包含不可覆盖的 JSON
清单和实际源码/config/数据视图快照 ZIP。checkpoint 只记录路径、大小和
SHA-256，不复制大权重。正式训练仍由用户手动启动。

每次训练结束再补齐：

- 最佳 checkpoint 的 SHA-256、EMA/raw 选择、optimizer updates、最佳验证窗口；
- 对应 Kaggle submission ref、CSV SHA-256、公榜分数；
- 该候选是否淘汰，以及被下一候选继承时的父子关系。

跨版本正式训练还必须在启动清单中记录父代 epoch/iter、继承/fresh optimizer
张量数、保留的可训练参数量占比、EMA 年龄和 FP16 scaler 状态。放行前执行两段
smoke：父代派生的 stateful 起点先更新并保存，再从子代 checkpoint 恢复更新；
旧参数 AdamW step 必须等于父代 step 加成功更新数。仅 model-only 加载成功、
loss 有限或子代内部 resume 成功，均不能证明没有重演 optimizer 失忆。

同时必须区分 runner attempted iterations、AdamW successful steps 和 AMP skips；
父 checkpoint 已保存 scaler 时，继承或 fresh 都要写出父值和理由。语义分支只
克隆权重并不自动授权克隆 moments。加载前必须运行版本对应的 fail-closed gate，
严格检查模型/EMA schema、optimizer name/order/shape/超参数、finite moments、
step 族、scaler、lineage、源码和上游 commit。MMCV `strict=False` 的无报错和
PyTorch optimizer group 长度相同都不能作为放行依据。

V5 的只读门禁示例：

```bash
PYTHONPATH="$PWD/src:$PWD/storage/upstream/Co-DETR" \
storage/envs/codino-v4-smoke/bin/python -m hod26.v5.load_gate \
  --contract configs/v5/state_load_contract.json \
  checkpoint \
  --checkpoint storage/v5/runs/co_dino_vitl_fdr_salience_fold0/latest.pth \
  --mode resume \
  --updates-per-epoch 1406
```

## Phase 2 单模型 CSV 流程

ranking 图片发布后先建立无标签视图：

```bash
storage/envs/deim-v2/bin/python -m hod26.v2.phase2 prepare-ranking \
  --image-folder /path/to/released/ranking/VIS
```

为最终胜出版本准备 test 与 ranking 两个推理配置。两者只能改变数据路径：
ranking 配置应指向生成的 `storage/v2/dataset/ranking.json` 和官方 ranking
图片目录。用同一个最终 checkpoint、同一种 EMA/raw 权重和相同后处理分别运行：

```bash
storage/envs/deim-v2/bin/python -m hod26.v2.infer \
  --config /path/to/final_test_infer.yaml \
  --checkpoint /path/to/FINAL.pth \
  --manifest storage/cache/test_manifest.json \
  --output storage/final/test.csv

storage/envs/deim-v2/bin/python -m hod26.v2.infer \
  --config /path/to/final_ranking_infer.yaml \
  --checkpoint /path/to/FINAL.pth \
  --manifest storage/cache/ranking_manifest.json \
  --output storage/final/ranking.csv

storage/envs/deim-v2/bin/python -m hod26.v2.phase2 combine \
  --test-csv storage/final/test.csv \
  --ranking-csv storage/final/ranking.csv \
  --output storage/final/phase2_submission.csv
```

每次推理会生成 `<csv>.audit.json`，其中含 checkpoint SHA-256。合并器会拒绝
两个分片来自不同 checkpoint、不同 EMA/raw 权重或不同架构，并重建全局连续
`id`。这保证 CSV 层面不会意外混用两个模型。

## 只在最终模型确定后执行

- 将胜出版本的源码、完整配置继承链和启动脚本整理到一个干净 Git commit；
- 固定 DEIM 上游源码 commit，选择随包附带源码或提供可校验的安装脚本；
- 锁定 Python/CUDA/PyTorch 与所有依赖，并在干净环境跑训练 smoke 和完整推理；
- 用最终输入尺寸实测参数量与 FLOPs；不得沿用 V3 候选的占位数字；
- 从最终权重重新生成 Phase 2 CSV，并核对 CSV、权重、源码包 SHA-256；
- README 写出从官方数据到 CSV 的逐条可复制命令；
- 由用户审核后再决定最终分支、commit、push、邮件或 Kaggle 上传。

当前阶段不应把 V3 分支或文档推送成“最终方案”。
