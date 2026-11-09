# reBotArm 精细动力学参数辨识

本目录把本机 `vendor/reBotArm_control_py`、机械臂 URDF 和 FIGAROH 串成一条可复现流程：原始反馈映射与重采样、刚体参数辨识、独立数据验证、物理一致性约束、URDF 导出，以及用辨识后模型计算重力补偿。

当前已完成离线链路、仿真真值回归、真实机械臂数据采集、独立静态重力验证和分阶段硬件保持测试。完整六轴动态辨识尚未通过最终验收；当前可部署成果是 `results/static_gravity_model.json` 中经过独立验证的静态重力修正，不应把未通过 verdict 的动态辨识结果用于控制。

注意：最新跨工作区静态结果为 `[1, 0.891304506, 1, 1, 1, 1]`，但它目前只通过离线独立验证和高刚度分阶段保持。低刚度测试暴露出直连 `motorbridge` 的跨轴反馈污染，因此当前状态不是“可直接拖拽”。方向配置仍为 `[1,-1,-1,1,1,1]`，没有改用旧工程中的历史方向表。详见 `results/gravity_hardware_validation_20261109.json`。

## 已确认的信息

- 模型关节顺序：`joint1` 到 `joint6`。
- 电机：前三轴 DM4340，后三轴 DM4310。
- URDF 力矩上限：`[27, 27, 27, 7, 7, 7]` N·m。
- `motorbridge` 提供位置、速度、估计力矩，但其力矩标定仍需用已知载荷验证。
- 历史代码中有两套冲突的方向表；当前已按现场确认选择 ROS 配置方向 `[1, -1, -1, 1, 1, 1]`，并在 `config/hardware_mapping.yaml` 中记录核验状态。零偏和绝对力矩比例仍保持名义值。
- 当前 STM32 程序是 USB-CAN 透明桥；旧 ROS 驱动的 64 字节聚合协议不能直接代表当前反馈格式。

## 环境

从本目录运行 FIGAROH 命令：

```bash
cd /home/nuc001/ibvs-migration/third_party/figaroh-examples/examples/rebotarm
/home/nuc001/ibvs-migration/.tools/miniforge3/envs/figaroh-dev/bin/python identification.py --help
```

硬件采集脚本使用 `vendor/reBotArm_control_py` 自己的 Python 3.10 环境，因为 `motorbridge` 不在 `figaroh-dev` 中。

## 1. 离线自检（已经通过）

```bash
python generate_simulated_data.py
python identification.py --no-plot --no-html-report --no-archive --no-verify
```

两条轨迹由不同随机种子生成，验证集不是训练集切分。名义 URDF 通过 Pinocchio RNEA 产生真值力矩。当前结果为 36 个基参数，训练和验证相关系数接近 1，CVXOPT 物理拟合的六个链节均可行。

## 2. 核验硬件映射

先逐轴、小幅、低速地确认“电机正方向”在 URDF 中是正还是负，并测量机械零位。将结果写入：

```text
config/hardware_mapping.yaml
```

映射定义为：

```text
q_urdf   = direction * q_motor + zero_offset_rad
tau_urdf = direction * torque_scale * tau_motor
```

角度和力矩必须同时乘方向，才能保持功率 `tau*dq` 的符号不变。再用已知质量、已知力臂做静态试验核验每轴力矩比例；完成后填写核验人和时间，并把 `verified` 改成 `true`。

## 3. 采集与转换

只读反馈采集器不会使能或发送运动指令：

```bash
# 用 reBotArm 的 Python 3.10 环境运行
python tools/record_feedback.py --output data/raw/train.csv --duration 60 --rate 100
```

它适合方向、零位和反馈质量检查。失能状态的电机估计力矩不一定有效，因此不能仅凭这种被动日志宣称完成动力学辨识。正式训练数据应由安全的激励运动控制器同步记录同一时钟下的 `q_motor` 和 `tau_motor`；不要让另一个进程同时占用串口。

转换为 FIGAROH 格式：

```bash
python tools/convert_raw_log.py data/raw/train.csv data/real/train.csv --rate 100
python tools/convert_raw_log.py data/raw/validation.csv data/real/validation.csv --rate 100
```

训练和验证必须是两次独立采集，不能把一条日志随机拆分。

## 4. 分阶段辨识

先只辨识刚体项：

```bash
python identification.py \
  --config config/rebotarm_unified_config.yaml \
  --data data/real/train.csv \
  --validation data/real/validation.csv \
  --asset-id rebotarm-001
```

检查报告中的每轴残差、条件数、参数不确定度和独立验证 RMSE。若残差随速度方向出现明显结构，再使用 `config/rebotarm_friction.yaml` 加入黏性摩擦、库仑摩擦和常值力矩偏置。只有残差明显随加速度变化、且激励确实覆盖足够加速度时，才进一步加入执行器惯量。

`config/real_acceptance_provisional.json` 采用额定力矩 3% 的每轴 RMSE 上限，只是初始验收线，应根据重复实验噪声和控制目标修订。

通过独立预测验收后才能导出：

```bash
python identification.py \
  --config config/rebotarm_unified_config.yaml \
  --data data/real/train.csv \
  --validation data/real/validation.csv \
  --verification-scope prediction \
  --acceptance-profile config/real_acceptance_provisional.json \
  --export-urdf results/rebotarm_identified.urdf
```

脚本会拒绝在没有独立验证、显式阈值或通过 verdict 时导出。输出采用直接物理约束拟合；这是因为 joint1 的轴向惯量只能以参数组合观测，普通逐链节重构虽可精确保持基参数，却可能产生单链节不可行惯量。

## 5. 重力补偿

先离线计算，不连接硬件：

```bash
python tools/gravity_compensation_identified.py \
  --urdf results/rebotarm_identified.urdf
```

确认模型、方向、零位和力矩指令比例后，才显式加 `--execute`。硬件模式要求映射已核验，默认将每轴前馈限制在额定力矩的 30%，并用 2 秒渐入。首次实验仍应托住机械臂、使用急停、降低限幅，并从保守姿态开始。

## CSV 数据契约

FIGAROH 输入固定为：

```text
t,q1,q2,q3,q4,q5,q6,tau1,tau2,tau3,tau4,tau5,tau6
```

- `t`：秒，严格递增并重采样到配置频率；
- `q*`：弧度，URDF 坐标系；
- `tau*`：N·m，关节侧、URDF 正方向；
- 速度和加速度由同一套滤波/差分流程从位置估计，避免混用不同时间基准。

