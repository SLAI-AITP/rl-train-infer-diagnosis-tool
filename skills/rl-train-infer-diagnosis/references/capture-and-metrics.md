# 采集契约与指标计算

本参考定义采集内容、数学公式、数值处理和脚本接口。诊断状态、选点顺序及后续动作统一由 [SKILL.md](../SKILL.md) 给出。

## 1. 场景与采集记录

有现成记录时保存其路径与来源；新采集时，从现有数据选择 prompt，使用当前推理入口运行 rollout，保存原始 token IDs、前缀、policy 快照标识和可取得的逐步 logprob，再让 train 加载同来源快照重放。记录新采集与历史场景的条件差别。训练比较阶段不更新参数，复测使用固定 token 轨迹。

| 对象 | 保存内容 |
|---|---|
| 权重 | policy 快照、加载／同步记录；涉及的转换、分片、expert／vocab 映射、adapter |
| 输入 | 原始 token IDs、前缀、目标 token／位置、BOS／EOS、模板、截断及输入来源；不经解码文本重新 tokenize 替代原 IDs |
| 评分口径 | 实际 temperature 及作用位置、原始／处理后 logits 或 logprob；相关 top-k／top-p、惩罚、bias、约束、stop |
| 上下文 | position、causal／padding／packing／滑窗、RoPE、序列边界、cache 与目标位置映射 |
| 执行条件 | batch／microbatch、TP／EP／PP、dtype、设备、prefill／decode、fusion、compile、dropout／路由噪声、router replay 原开关与作用范围 |
| 代码来源 | 入口／命令、当前可得的版本／commit、运行时文件与函数、目标 kernel、插桩／适配变换 |

按采集阶段已有信息填写，未知字段明确标记；需要的细节可在后续对应实验中补齐。完整 kernel 清单和逐参数哈希不属于最小输入格式。

固定 token 后，只作用于采样的温度不参与此前 hidden 的计算。记录 logprob 具体使用温度处理前还是处理后的 logits；temperature=0 的生成接口不按 logits 除以 0 实现。原始 decode 与 full-sequence prefill 分别标识，记录各自 cache 路径。

## 2. 逻辑 token 与特征对齐

为位置分配稳定且唯一的 `token_key`，如 `sample-0007:position-0123`，另存实际 token ID 与目标 token ID。分别保存：

- `activation_valid_mask`：用于 hidden 比较的有效输入位置，排除 padding。
- `logprob_valid_mask`：评分位置，通常为有效 response token。

记录是否包含 prompt／EOS，以及两类 mask 的对应映射。同类比较使用相同有效集合，不自动取两端交集。逆转 packing／dispatch 重排，重建逻辑 head、expert、特征和 rank 映射。TP shard 先恢复完整特征；DP／副本仅计一次逻辑 token。不同内部表示需有显式语义映射。

每份数据记录边界、原 shape／dtype／布局、导出变换、token ID 校验、mask、sample／request／step／rank、状态来源。布局和实际入口转换的代码模式见 [插桩方法](instrumentation-guide.md)。

## 3. cosine 与方向差异

有效 token 集为 V，每个 token 对应完整逻辑特征 a_t、b_t：

```text
c_t = dot(a_t,b_t) / (norm(a_t) * norm(b_t))
C(a,b) = sum(c_t, t in V) / |V|
Dcos(a,b) = 1 − C(a,b)
```

跨 batch／sample／DP rank 聚合 cosine 总和与有效 token 数。保留 `n_valid`、逐 token 值和最差位置；不使用整张量一次 cosine、各 rank 均值的无权平均或 shard cosine 平均。

至少提升到 FP32 后计算，细微差异使用 FP64 累加；减法前先提升精度。随附脚本以 FP64 对每行缩放、归一化并点积，避免大／小数值溢出。保留未舍入结果及 Dcos。

有效零范数的 cosine 未定义；双零也不设为 1。统计零范数、NaN／Inf 的位置与数量。完整集合含未定义值时整体 C 置空，另存有限非零子集统计及其 token 集；padding 不属于有效集合。

## 4. 输入／输出指标与增长量

对每个层或子模块 l，xT_l、xI_l 是两端实际输入，yT_l、yI_l 是对应输出：

```text
C_in,l  = C(xT_l, xI_l)
C_out,l = C(yT_l, yI_l)
D_in,l  = 1 − C_in,l
D_out,l = 1 − C_out,l
G_l     = D_out,l − D_in,l = C_in,l − C_out,l
```

G 保留正负号，不取绝对值，不计算相对增长比。输入、输出使用同一有效 token 集，分别完成两端语义对齐。仅当前一采集输出就是本操作实际输入时可直接引用；遗漏的 norm、cast、residual、通信或多分支输入显式记录。缺失或不相容的输入使对应 G 置空并记录原因。

输出表字段：`scope、operation、input_point、output_point、C_in、C_out、Dcos、G、n_valid、data_status、evidence_path`。有重复时附各次原值、G 中位数和范围。

## 5. 全前序层统计

输入为当前路径上相同语义位置的层输出 cosine 序列 C_1,…,C_L，以及主流程指定的 `t、min_prefix_layers`。层号从 1 开始，当前层 l 的前序数 n=l−1：

```text
mu_prefix,l     = sum(C_i, i=1,…,l−1) / n
sigma²_prefix,l = sum((C_i−mu_prefix,l)², i=1,…,l−1) / (n−1)
sigma_prefix,l  = sqrt(sigma²_prefix,l)
lower_l         = mu_prefix,l − t * sigma_prefix,l
```

采用样本标准差 ddof=1。计算只包含全部前序层，不包含当前层、后续层，不使用滑动窗口或删去历史低值。σ 是层均值之间的标准差，不是 token 间标准差或均值标准误。

`prefix_ready` 表示前序数达到给定最小值、前序完整且指标有效、位置语义与 token 集可比。n<2 时样本标准差和 lower 置空；其他条件不足时保留可计算原值但标记对应原因。embedding、层输入、final norm、logits 分别记录，不混入 decoder 输出序列。

FP64 使用中心化的方差计算，保留原精度结果。σ=0 时 lower=mu，不计算除以 σ 的 z-score；另记 `near_constant` 及计算精度。各次复测分别输出整条前序统计，不跨运行混成一条序列。

输出字段：`n_prefix、mu_prefix、sigma_prefix、t、lower、prefix_ready、prefix_missing、near_constant`。例如输入统计 mu=0.99990、sigma=0.00001、t=3，计算结果 lower=0.99987。

## 6. 辅助与端点指标

辅助 hidden 指标保留逐 token 值及 mean／P95／P99／max：

```text
relative_l2_t = norm(b_t − a_t) / norm(a_t)    # a 为 train 参考
norm_ratio_t  = norm(b_t) / norm(a_t)
max_abs_error = max(abs(b_t − a_t), over valid tokens and features)
```

减法前提升精度；零分母输出未定义，不静默添加 epsilon。与脚本接口不同的有限差分增益见重放参考中的单独公式。

目标 token 的自然对数概率使用相同评分口径：

```text
delta_logprob_t = logp_train,t − logp_infer,t
delta_p_t       = exp(logp_train,t) − exp(logp_infer,t)
E_logp         = mean(abs(delta_logprob_t), t in scoring set)
```

保留有符号均值、绝对值 mean／P95／P99／max、token 数和异常位置。exp 下溢时保留原 logprob；接口缺失的目标概率记缺失，不填 0。top-k 截断概率未提供完整词表分布时，不计算精确全分布 KL。记录 logits 的 vocab 映射、log-softmax 阶段和 target gather 索引。

## 7. 重复采集与指标精度记录

在固定权重、token、状态及执行条件下保存各次 T、I 输出。分别计算 T→T、I→I 的重复间 cosine／可选辅助误差，以及每次 T→I 指标；保留样本数、原值、中位数、范围、测量 dtype 与舍入方式。插桩、重放路径或执行条件改变时使用对应的记录。端点无插桩／有插桩输出也按相同评分集合保存。测量容差的取值与依据写入 manifest。

## 8. 随附脚本接口

离线参考环境使用 Python 3.9–3.12 和 NumPy 1.26.4，安装本 skill 的 `requirements.txt` 可使用固定依赖；其他组合需另行验证。脚本路径相对于本 skill 目录。输入是一对已对齐的 NPZ，每份必须包含：

| 键 | 类型与 shape | 含义 |
|---|---|---|
| `values` | 浮点 `[N,D]` 或 `[N]` | hidden 完整特征或目标 logprob |
| `token_keys` | Unicode `[N]`、非空唯一 | 相同顺序的位置标识，含 padding 占位 key |
| `valid_mask` | bool `[N]` | 当前类别的有效集合 |

两端 keys、mask、shape 要一致；适配层负责映射，不使用 object／pickle。BF16 等不直接支持的格式无损提升为 FP32 并记录原 dtype。

```bash
python scripts/compare_tensors.py --kind hidden --train train_layer.npz --infer infer_layer.npz --output layer_metrics.json --per-token layer_cosines.npz
python scripts/compare_tensors.py --kind hidden --train train_layer.npz --infer infer_layer.npz --output layer_aux.json --auxiliary --per-token layer_aux_tokens.npz
python scripts/compare_tensors.py --kind logprobs --train train_logp.npz --infer infer_logp.npz --output logprob_metrics.json
python -m unittest discover -s scripts -p 'test_*.py'
```

`cosine_mean` 对应 C，`one_minus_cosine_mean` 对应 Dcos；`cosine_defined_subset_mean` 是有限非零子集均值。`status=ok` 是数据／计算状态，有效零范数或非有限值会使完整 C 置空。辅助输出为 `relative_l2_train_denominator、infer_over_train_norm、max_abs_error`；`abs_delta_logprob.mean` 对应 E_logp。增长量、前序统计及后续实验表由调用方按本参考整理，脚本不负责候选排序或诊断状态。

输出 `schema_version=2`。hidden 的逐 token NPZ 始终包含 `cosine、token_keys、valid_mask`；指定 `--auxiliary` 时再包含 `relative_l2、norm_ratio、max_abs_error` 三个 `[N]` 数组。`max_abs_error` 数组是每个 token 特征上的最大绝对误差；JSON 同名字段仍是所有有效位置的全局最大值。数组保持原 token 顺序，padding 为 NaN；零分母和非有限输入对应的未定义指标不填 0。汇总仅统计有效 mask，未定义计数不含 padding。

与 schema 1 相比，汇总字段保留，增加辅助逐 token 数组；CLI 默认 `--path-mode omit`，不再自动写入输入绝对路径。需要本地追溯时显式使用 `--path-mode absolute`，结果才含 `train_path、infer_path`。NPZ 输入格式未变。分享前仍需检查 token key、数据内容及其他实验记录。

`--chunk-rows` 只控制计算分块；输入 NPZ 的两端数组仍会读取到内存，大 trace 可按边界拆分文件。

## 一手参考

- [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)：数值精度与归约背景。
- [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)：batch 与确定性实现资料。
- [NIST: What are Control Charts?](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm)：均值与标准差界限的背景。
