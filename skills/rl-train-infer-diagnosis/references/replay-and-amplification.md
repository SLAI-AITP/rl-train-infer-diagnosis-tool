# 重放与扰动实验：输入、运行方法和指标

本参考给出输入包、四组运行、控制实验及指标公式。结果判读、证据状态和下一步统一见 [SKILL.md](../SKILL.md)。

## 1. 输入包与执行条件

为当前受测区间保存完整实际输入：浮点张量、residual／多分支状态、position／mask、长度、KV 或压缩历史、路由／dispatch 元数据、随机状态（若有）、原 batch／shape 和并行／kernel 条件。权重单独固定并记录。

- 测试包含 router 的整个 MoE 时，路由由内部原生代码按原配置产生；测试 router 之后的 expert／dispatch 时，路由结果属于该子区间的外部输入。记录 router replay 原开关、重放字段和覆盖范围。
- Attention 的全序列上下文与 cache 记录同一逻辑历史。物理格式转换保存适配代码、转换前后数值及位置元数据；不只替换当前 hidden 而省略历史输入。
- 多 token 运算保留相关 batch、expert 分组和归约调度条件；缩小 shape 的运行单独记录配置。
- 同一规范输入包分发到两端，记录实际入口值、cast 和布局变换。布局转换保持数值与逻辑映射；入口 dtype 转换的前后值分别保存。

使用独立快照，保留正确 stream 依赖；多次重放前恢复有副作用的状态。输入包来源、固定字段、改变字段、缺失字段和人工构造的反事实状态均写入实验记录。

## 2. 四组运行及自重放指标

T、I 为两端当前实现；xT、xI 为当前轮 trace 的对应实际输入包；yT、yI 为该轮原始输出。固定权重与指定条件，得到：

| 输入 | 训练实现 T | 推理实现 I |
|---|---|---|
| xT | a=T(xT) | b=I(xT) |
| xI | c=T(xI) | d=I(xI) |

每次运行记录实际输入／状态、输出、rank／request／step 与 kernel 路径。C 的计算使用 [采集约定](capture-and-metrics.md) 中同一有效 token 集。

| 主流程字段 | 计算公式 | 数据对象 |
|---|---|---|
| R_T | C(a,yT) | 训练端自重放与自身原 trace |
| R_I | C(d,yI) | 推理端自重放与自身原 trace |
| S_T | C(a,b) | 固定训练侧输入的跨实现输出 |
| S_I | C(c,d) | 固定推理侧输入的跨实现输出 |
| C_in | C(xT,xI) | 被研究的浮点输入 |
| P_T | C(a,c) | 固定训练实现、改变输入的输出 |
| P_I | C(b,d) | 固定推理实现、改变输入的输出 |
| C_natural | C(a,d) | 两端各自原始输入经过各自实现后的输出 |

保存输入包其他字段的控制方式；多种浮点输入可分别计算 C_in 并标注目标变量。复用已有记录时显式保存字段映射，不仅凭名称判断计算语义。

自重放验收同时检查方向和幅值：除 R_T／R_I 外，记录相对 L2、范数比和最大绝对误差，并按声明容差验证；零范数处以绝对误差和状态检查补充，不仅依赖 cosine。同输入 S_T／S_I 被判为接近一致时也检查幅值误差。重复运行记录每次指标及范围，保留无插桩／有插桩路径控制。技术排查项包括状态恢复、异步同步、in-place 覆盖、导出 cast、kernel dispatch、布局和随机状态。

## 3. 实际差异方向的扰动输入

对选定浮点输入定义：

```text
delta = xI − xT
x_alpha = xT + alpha * delta
alpha 示例：0、0.25、0.5、1
```

alpha=0 和 1 直接使用保存的真实 xT、xI，避免构造舍入改变端点。中间点在足够精度下构造后转实际入口 dtype，记录最终收到的值及不同 alpha 是否量化为同一输入。离散 token、位置和 expert IDs 不作线性插值。

两端接收相同规范 x_alpha；固定其余输入、状态与执行条件。记录少量其他真实输入的相同测量。多输入／有状态边界可以分别固定 cache 测 hidden、固定 hidden 测语义可用的 cache；保存每组完整变量控制清单及状态可构造性。

每个 alpha 运行 u_alpha=T(x_alpha)、v_alpha=I(x_alpha)，保存：

```text
S_alpha    = C(u_alpha, v_alpha)
C_in_alpha = C(xT, x_alpha)
P_T_alpha  = C(u_0, u_alpha)
P_I_alpha  = C(v_0, v_alpha)
```

u_0、v_0 来自本组 alpha=0 的完整输入包；只有该输入包及执行条件与原 xT 重放一致时，才可复用 a、b。仅插值某个浮点字段时，alpha=1 仅保证该字段等于 xI 的对应字段，其余受控状态另行记录。重复／随机执行时另存实际运行编号，不混淆不同次基准。若采集离散选择，保存 expert 集合、权重、margin 和 dispatch 索引，以及它们对应的 alpha、token、step。

## 4. 可选幅值指标与代数关系

幅值响应可计算归一化有限差分，使用独立字段 `gain_norm`，不与层间增长量 G 混用：

```text
gain_norm = [norm(F(x+delta)−F(x)) / (norm(F(x))+eps)]
            / [norm(delta) / (norm(x)+eps)]
```

固定 norm 的 token／feature 范围、eps、dtype 与尺度。alpha=0 或实际扰动为零时增益置空；记录近零分母。cosine 只度量方向，gain_norm 另含幅值信息。

在输出空间与布局可加的条件下：

```text
d − a = (b − a) + (d − b)
d − a = (c − a) + (d − c)
```

向量项可能抵消；norm 和 cosine distance 不具有上述可加性。这些表达式不用于将距离归一化成可相加的贡献率。
