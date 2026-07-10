# Ledger-WAM / LingBot-VA 项目说明

本项目基于 LingBot-VA 代码实现，并在其视频-动作世界模型基础上加入
Ledger-WAM 能力：结构化因果账本、因果债务估计、自愈式修复动作选择、
局部逻辑回滚、反事实动作监督和在线修复执行确认协议。

Ledger-WAM 是 opt-in 设计。默认 LingBot-VA 配置仍保持原行为；只有启用
`ledger_enabled=True` 或使用 `ledger_*` 配置时，才会进入 Ledger-WAM 训练和
推理路径。

## 项目状态

当前仓库已经包含以下能力：

- LingBot-VA 原始视频-动作联合生成、训练和服务端推理代码。
- 结构化 Ledger 神经头：claim slot、实体、关系、前置条件、效果、证据、
  不确定性、依赖、可观测性、重要性、因果债务、回滚阶段、修复动作等预测。
- 单调因果债务估计：低置信度、高不确定性、高下游依赖、高修复成本、
  低可观测性都会增加债务。
- 反事实动作建模：共享 Siamese transition head，避免 factual/counterfactual
  使用不同头造成虚假 margin。
- 自愈规划器：基于全局风险、修复收益、动作成本和任务风险选择 task action、
  repair action 或 local logical rollback。
- 在线因果账本运行时：证据融合、依赖传播、逻辑回滚、运行时状态序列化。
- 物理修复执行握手协议：服务端发出 repair chunk 后，控制端必须在下一次
  `compute_kv_cache=True` 请求中回传 `repair_execution_ack`。
- RoboTwin 和 LIBERO 客户端适配；RoboTwin 路径、action 维度、相对位姿处理
  已做显式配置与错误提示。
- Ledger 相关单元测试、示例 sidecar 标注和评估指标。

尚未完整闭环的部分：

- VLABench 需要单独新增 benchmark runner / adapter。本仓库目前没有完整
  `evaluation/vlabench` 入口。
- 完整论文级复现实验还需要真实数据、因果标注、已训练 Ledger checkpoint、
  benchmark 环境和机器人专用 repair executor。
- 当前轻量验证覆盖单测和编译，不等价于 CUDA/GPU 端到端训练或仿真评测。

## 目录结构

```text
.
├── README.md                         # 上游 LingBot-VA 说明，保留原信息
├── README_CN.md                      # 本文件，项目中文详细说明
├── LEDGER_WAM.md                     # Ledger-WAM 数据契约、训练与在线协议细节
├── INSTALL.md                        # 安装补充说明
├── requirements.txt                  # 主要 Python 依赖
├── pyproject.toml                    # Python 包配置
├── example/
│   ├── ledger_annotations.example.jsonl
│   ├── robotwin/
│   ├── libero/
│   ├── franka/
│   └── demo/
├── evaluation/
│   ├── libero/                       # LIBERO 客户端/启动脚本
│   └── robotwin/                     # RoboTwin 客户端/启动脚本
├── script/
│   ├── run_va_posttrain.sh           # 训练入口
│   └── run_launch_va_server_sync.sh  # i2va / server 启动入口
├── tests/                            # Ledger 单元测试
└── wan_va/
    ├── configs/                      # LingBot-VA 与 Ledger-WAM 配置
    ├── dataset/                      # LeRobot latent dataset 与 Ledger sidecar schema
    ├── ledger/                       # Ledger-WAM 神经头、运行时、协议、指标
    ├── modules/                      # Transformer / VA 模型结构
    ├── train.py                      # 训练主逻辑
    └── wan_va_server.py              # 在线推理服务端
```

## 环境要求

推荐环境：

```text
Python >= 3.10, < 4.0
CUDA 12.6
PyTorch 2.9.0
torchvision 0.24.0
torchaudio 2.9.0
diffusers 0.36.0
transformers 4.55.2
lerobot 0.3.3
```

基础安装：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

开发/测试依赖：

```bash
pip install pytest black flake8 isort mypy huggingface-hub[cli]
```

如果使用 `pip install .`：

```bash
pip install .
pip install .[train]
```

## 模型与数据

LingBot-VA checkpoint 和数据集可参考原始 `README.md` 中的 HuggingFace /
ModelScope 链接。常用数据包括：

- `robbyant/lingbot-va-base`
- `robbyant/lingbot-va-posttrain-robotwin`
- `robbyant/lingbot-va-posttrain-libero-long`
- `robbyant/robotwin-clean-and-aug-lerobot`
- `robbyant/libero-long-lerobot`

训练数据路径建议通过环境变量配置：

```bash
export LEDGER_WAM_DATASET_PATH=/path/to/your/dataset
```

RoboTwin 评测环境路径建议通过环境变量或命令行配置：

```bash
export ROBOWIN_ROOT=/path/to/RoboTwin
```

或在客户端启动时传入：

```bash
python -m evaluation.robotwin.eval_polict_client_openpi \
  --robowin_root /path/to/RoboTwin \
  --config policy/ACT/deploy_policy.yml \
  --overrides ...
```

## 关键配置

Ledger-WAM 通过配置开关启用：

```python
ledger_enabled = True
```

常用配置字段：

```text
ledger_annotation_path          Ledger sidecar JSON/JSONL 路径
ledger_strict                   是否严格校验 sidecar
ledger_max_claims               每段最多 claim slot 数
ledger_max_counterfactuals      每段最多反事实动作数
ledger_action_dim               Ledger 标注动作维度
ledger_debt_threshold           claim 级债务阈值
ledger_global_risk_threshold    全局风险阈值
ledger_confidence_threshold     证据置信阈值
ledger_repair_cost_weight       修复动作成本权重 beta
ledger_repair_risk_weight       修复任务风险权重 gamma
ledger_repair_catalog           离散修复技能目录
ledger_allow_random_head        是否允许加载未训练 Ledger head，仅调试用
ledger_allow_prompt_repair_fallback 是否允许 prompt recovery fallback，仅调试用
```

RoboTwin / Franka 相对位姿转换通过配置声明：

```python
relative_action_pose_slices = ((0, 7), (8, 15))  # RoboTwin
relative_action_pose_slices = ((0, 7), (7, 14))  # Franka
```

每个 slice 必须是 `[x, y, z, qx, qy, qz, qw]` 的 7 维 pose。LIBERO 和 demo
默认不启用该转换，避免改变原始动作语义。

## Ledger 标注格式

Ledger post-training 使用 sidecar 标注文件。默认路径：

```text
<LeRobot dataset root>/meta/ledger_annotations.jsonl
```

也可以通过 `ledger_annotation_path` 指定。

最小示例见：

```text
example/ledger_annotations.example.jsonl
```

一条典型记录：

```json
{
  "key": "12:30:90",
  "episode_index": 12,
  "start_frame": 30,
  "end_frame": 90,
  "claims": [
    {
      "claim": 1.0,
      "claim_type": 1,
      "subject": 3,
      "object": 7,
      "relation": 9,
      "precondition": 4,
      "effect": 2,
      "evidence": 0.8,
      "uncertainty": 0.2,
      "dependency": 0.9,
      "observability": 0.4,
      "repair_cost": 0.35,
      "importance": 1.0,
      "debt": 0.75,
      "rollback": 4,
      "repair_action": 1,
      "post_repair_debt": 0.15
    }
  ],
  "dependency_edges": [
    {"source": 0, "target": 1, "weight": 1.0}
  ],
  "counterfactual_actions": [
    {"action": [0.0, 0.1], "delta": [1.0]}
  ]
}
```

说明：

- `key` 格式为 `episode_index:start_frame:end_frame`。
- 字段可以部分缺失，缺失标签会被 mask，不会当作负样本。
- `dependency_edges` 表示有向依赖边；也可以使用 dense `dependency_matrix`。
- `counterfactual_actions[].action` 是模型对齐后的当前动作向量。
- `repair_action` 使用 `ledger_repair_catalog` 的 list index，`id` 必须等于列表位置。
- `rollback` 使用固定 `ledger_rollback_stage_ontology`，不是动态 chunk index。

更完整的数据契约见 `LEDGER_WAM.md`。

## 训练

基础 post-training：

```bash
CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh
```

Ledger-WAM 训练：

```bash
CONFIG_NAME=ledger_robotwin_train \
  bash script/run_va_posttrain.sh \
  --ledger-annotation-path /path/to/ledger_annotations.jsonl \
  --ledger-strict
```

可替换配置：

```text
ledger_robotwin_train
ledger_libero_train
ledger_demo_train
```

训练逻辑会同时优化：

- LingBot-VA 视频/action flow matching loss。
- Ledger claim presence / truth / relation / entity / precondition / effect。
- evidence、uncertainty、dependency、observability、importance、debt。
- directed dependency matrix。
- rollback stage。
- repair action、repair cost、post repair debt。
- counterfactual transition 和 normalized L1 margin。
- recurrent slot identity consistency。
- paper 里的 repair reward：
  `debt_reduction - beta * action_cost - gamma * task_risk`。

注意：原始 latent segment 中包含未来目标，因此 Ledger head 默认只看第一帧
观测/action，避免标签或未来信息泄漏。

## 推理服务

RoboTwin 服务端：

```bash
bash evaluation/robotwin/launch_server.sh
```

RoboTwin 客户端：

```bash
export ROBOWIN_ROOT=/path/to/RoboTwin

task_name="adjust_bottle"
save_root="results"
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}
```

LIBERO：

```bash
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh
```

i2va 生成：

```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

## 在线 Ledger 协议

重置：

```python
client.infer({
    "reset": True,
    "prompt": "put the cup in the box",
    "planning_checkpoints": [
        {"checkpoint_id": "task_start", "cursor": 0},
        {"checkpoint_id": "grasp", "cursor": 3, "subgoal": "grasp"}
    ]
})
```

普通观测可附带外部因果更新：

```python
client.infer({
    "obs": observation,
    "ledger_claims": [serialized_claim],
    "ledger_dependencies": [["grasp", "transport"]],
    "ledger_evidence": [
        {
            "claim_id": "grasp",
            "source": "tactile_sensor",
            "polarity": "contradicts",
            "strength": 0.9,
            "timestamp": 12
        }
    ]
})
```

当 planner 选择 repair action 时，生产环境必须提供离散 repair skill 到连续动作
chunk 的映射：

```python
request["repair_action_chunks"] = {
    "lift_test": lift_test_action_chunk,
    "local_rollback": local_recovery_action_chunk
}
```

动作 chunk 形状：

```text
[used_action_channels, frame_chunk_size, actions_per_frame]
```

如果没有可执行 repair chunk，服务端会 fail-closed：

```python
{
    "action": None,
    "requires_repair_action": True,
    "repair_action_id": "lift_test",
    "repair_instruction": "..."
}
```

当 repair chunk 已发出，控制端必须在下一次 cache update 中确认执行：

```python
client.infer({
    "compute_kv_cache": True,
    "obs": key_frame_list,
    "state": executed_action,
    "repair_execution_ack": {
        "action_id": response["repair_action_id"],
        "execution_id": response["repair_execution"]["execution_id"],
        "success": True
    }
})
```

在收到 ack 前，服务端不会继续发第二个 repair action。

## 评估指标

`wan_va/ledger/metrics.py` 提供：

```text
claim_root_cause_metrics
top_k_accuracy
debt_calibration_metrics
rollback_metrics
repair_metrics
local_rollback_metrics
compute_ledger_metrics
```

这些指标覆盖：

- root-cause claim 定位。
- repair action top-k。
- debt calibration / ECE。
- rollback stage accuracy / distance。
- repair 后债务下降。
- local rollback 频率。

## 验证

轻量验证：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q \
  wan_va tests evaluation script

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

最近一次本地验证结果：

```text
63 tests OK
关键文件 compileall OK
```

完整 GPU 验证还需要：

- Python 3.10 环境。
- CUDA / PyTorch / flash-attn 正确安装。
- 真实 checkpoint。
- 真实 LeRobot 数据集和 latent 文件。
- RoboTwin / LIBERO 仿真环境。

## 常见问题

### 1. RoboTwin 客户端找不到环境

设置：

```bash
export ROBOWIN_ROOT=/path/to/RoboTwin
```

或运行时传：

```bash
--robowin_root /path/to/RoboTwin
```

也可以在 deploy config 中加入：

```yaml
robowin_root: /path/to/RoboTwin
```

### 2. action channel 数不是 14 或 16

RoboTwin 客户端目前支持：

```text
14: left xyz+rpy+gripper, right xyz+rpy+gripper
16: left xyz+quat+gripper, right xyz+quat+gripper
```

其它 layout 需要新增环境专用转换函数，不能直接复用当前 RoboTwin `ee` action
转换。

### 3. Ledger checkpoint 加载失败

如果 `ledger_enabled=True`，服务端会检查 checkpoint 的 transformer config 是否
包含训练过的 Ledger head。调试时可设置：

```python
ledger_allow_random_head = True
```

但这只适合连通性调试，不能代表论文功能效果。

### 4. `attn_mode` 报错

训练和推理需要不同设置：

```text
training:  "flex"
inference: "torch" 或 "flashattn"
```

该值通常在 checkpoint 的 `transformer/config.json` 中。

### 5. repair action 没有执行

服务端只选择离散 repair skill，不会凭空知道机器人怎么执行。生产环境必须提供：

```python
repair_action_chunks
```

并在执行后回传：

```python
repair_execution_ack
```

### 6. VLABench 为什么没有直接命令

VLABench 是独立 benchmark 环境，需要新增 `evaluation/vlabench` adapter，把它的
observation/action API 映射到本项目的服务器协议。当前仓库没有完整 VLABench
runner，因此不能把论文实验的 VLABench 部分视为已经一键复现。

## 开发建议

新增数据集时优先确认：

- `obs_cam_keys` 是否和数据字段一致。
- `used_action_channel_ids` 是否和模型输出一致。
- `inverse_used_action_channel_ids` 是否覆盖了所有模型 action 维度。
- `norm_stat.q01/q99` 是否和训练数据一致。
- 是否需要 `relative_action_pose_slices`。
- Ledger sidecar 的 `action_dim` 是否和 `ledger_action_dim` 一致。
- repair catalog 的 `id` 是否严格等于 list index。

新增 benchmark adapter 时建议拆成三层：

- observation formatter：把环境观测转成本项目 server 需要的 `obs` 字典。
- action executor：把 server 返回 action 转成环境动作。
- repair executor：把 Ledger 离散 repair skill 映射为可执行 action chunk。

## License 与引用

本项目保留 LingBot-VA 原 Apache-2.0 License。引用 LingBot-VA 或 Ledger-WAM
相关工作时，请同时参考原始 `README.md` 和论文说明。
