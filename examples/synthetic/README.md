# 合成示例

在仓库根目录运行 `python examples/synthetic/run_demo.py --output outputs/synthetic`。只需要 NumPy，数据由脚本生成，输出目录必须不存在。

案例一仅将第一个 token 的特征扩大两倍。预期 cosine 约为 1，逐 token relative L2 为 `[1, 0]`、norm ratio 为 `[2, 1]`、max absolute error 为 `[2, 0]`。可按 [CLI 接口说明](../../skills/rl-train-infer-diagnosis/references/capture-and-metrics.md)重新读取 `scale/train.npz` 和 `scale/infer.npz`。

案例二是二维无状态函数：训练侧 source 保持输入；推理侧 source 增加 `[0, 0.02]`；两端的 amplifier 均按元素乘以 `[1, 20]`。该构造使方向差异在 amplifier 处增大，但四组重放显示两端 amplifier 在同输入上相等。因此最大 G 对应放大器，不能直接当作误差源。替换当前推理输入上的 source、撤销替换，并使用另一组输入，展示端点验证。

`R_T／R_I` 将独立重放调用与此前的自然调用比较。这个确定性无状态函数无法验证真实引擎的 cache、worker、异步同步或随机状态恢复。

| 文件 | 内容 |
|---|---|
| `scale/`、`source/`、`amplifier/` | 两端输入文件、汇总指标、逐 token 指标 |
| `replay_a.npz` 至 `replay_d.npz` | amplifier 四组运行输出 |
| `replay.json` | 方向与幅值重放指标 |
| `chain/` | 相同目标的两端自然对数概率 |
| `summary.json` | G、replay 和干预端点汇总 |
| `run_manifest.json` | 合成来源、函数差别、评分语义和容差 |
| `report.md` | 从实测结果生成的中文说明 |

这些数据检验数值工具和合成实验关系，不代表模型准确率、真实框架支持或长期 RL 收敛改善。示例不赋予 `MAIN_CAUSE_CONFIRMED` 等真实诊断状态。
