# 层内观测点与算子实验方法

本参考提供内部采集点映射、观测字段和可控实现变体。候选选择、递归循环与诊断结论统一见 [SKILL.md](../SKILL.md)。实际 hook、包装器、输入注入和分片处理见 [插桩方法](instrumentation-guide.md)。

## 1. 层内观测记录

沿当前 forward 的真实依赖关系，将两端对应点映射到操作的实际输入和输出。记录：

```text
scope / parent_scope / operation
input_point / output_point / branch / call_index
source_file / function / kernel / version
input_snapshot / output_snapshot / state_snapshot
C_in / C_out / G / n_valid / data_status / evidence_path
```

同一父输入运行中，保存各子模块实际收到的输入，而不是把父层输入复用为所有子模块输入。residual／多分支输入分别保存，记录合并前分支和合并后输出。数据对齐、C 与 G 的公式见 [采集约定](capture-and-metrics.md)。已有符合当前输入、配置和映射的指标可直接引用；只缺指标时对已保存张量离线补算。

## 2. 各区域的观测点

按实际实现配置探针，不要求所有模型有相同模块结构。

| 区域 | 采集点／字段 | 实验准备 |
|---|---|---|
| Norm／Residual | norm 输入、平方／归约、归一化、scale、残差分支、相加前后 | 记录 epsilon、cast、累加精度、残差流和融合顺序 |
| Attention | Q/K/V 或等价表示、位置处理、score、mask／softmax、V 聚合、输出投影 | 完整上下文／cache、位置偏移、head 映射、scale、输入／输出 layout |
| MoE router | router 输入、scores、bias、top-k IDs、权重、分组边界 | 全局 expert ID、tie-break 规则和 logits／概率的具体定义 |
| MoE dispatch／combine | dispatch 前后、各 expert 输入输出、权重应用位置、合并前后 | 原生 permutation／unpermutation、token 与 expert 映射、分片与归约顺序 |
| Dense／expert FFN | gate／up、激活、逐元素乘、down | activation 语义、cast、feature 分片、权重与融合参数 |
| GEMM／量化 | 输入／权重变换、scale、乘加、输出 cast | 源权重与执行权重、accumulator、舍入、饱和、量化粒度 |
| 通信 | collective 前后、各 rank 对应分片、合并输出 | 参与 rank／顺序、归约 dtype、逻辑布局 |
| 输出评分 | final norm、LM head、vocab 分片、logsumexp、temperature、gather | 词表映射、目标偏移、处理前后口径、完整归约范围 |

路由数据按 token 保存选中 expert 集合与权重；选中集合的比较使用集合一致率或有序 ID 一致率，并明确是否考虑顺序。margin 可记录第 k 与第 k+1 个候选 score 的差，说明 score 是原始 logit、校正后 score 还是概率。输入、token 集和计算阶段写入字段元数据。

## 3. 实现变体与 fused 区间的记录

保存实际 dispatch 的 kernel 名、入口参数、shape、dtype 和运行时调用位置。只有外部输入／输出可见的 fused kernel，观测点标为 fused 输入和 fused 输出，内部点标记不可采集。

拆分实现、禁用 compile、参考算子或更高精度运行分别作为有名称的实验变体，保存：

- 原路径与变体的具体代码、输入／状态和执行条件差别。
- 两端相同输入下的可见输出及原路径／变体各自端点。
- 实际 fusion／kernel／graph 是否改变及相关日志。
- 高精度参考的 epsilon、scale、位置／mask、舍入、归约和输出转换语义。

变体不覆盖原始观测文件。局部改动明确列出唯一改变项；整模块替换或多项改变记录全部修改集合。机制跨层重复时，分别标记单实例与多实例运行的全局层号。

## 4. 算子实验记录模板

```text
experiment_id / scope / selected_operation
假设对应的实现差别
固定：输入、状态、权重、shape、布局、其余执行条件
改变：具体代码／配置项；其他伴随变化
观测：同输入输出、内部点指标、对应完整轨迹端点
原始／变体／撤销的命令和产物路径
```

模板只描述实验实施和观测字段，不预设某种精度、路由或 kernel 差异是当前问题的原因。
