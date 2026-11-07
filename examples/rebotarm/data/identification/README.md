# Identification data

规范输入列为：

```text
t,q1,q2,q3,q4,q5,q6,tau1,tau2,tau3,tau4,tau5,tau6
```

单位分别为秒、弧度、N·m；所有关节均处于 URDF 坐标系。训练和验证必须来自两次独立采集。原始 motorbridge 日志不要直接放入辨识器，先用 `../../tools/convert_raw_log.py` 应用已经核验的硬件映射并等周期重采样。

