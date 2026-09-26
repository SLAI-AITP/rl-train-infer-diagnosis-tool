# 插桩与重放：如何修改实际运行代码

当 agent 需要取得逐层输出、选择 forward 插桩位置或将保存输入接入另一实现时使用。本参考提供通用采集和重放模式；代码片段需要项目适配，不代表已验证的框架集成。

本参考仅提供改代码、导出和技术校验方法。诊断结论及指标对应的下一步统一见 [SKILL.md](../SKILL.md)。

示例需要按当前代码适配 Python 环境、模型对象、worker 入口、forward 参数、token 映射和布局。层号、流数、hidden 大小、token 数及 TP／EP 数量均由当前模型决定。eval、eager、禁用 cache／chunked prefill 和 router replay 开关属于实验变量，不是诊断必选配置。

## 1. 从一个能核验的采集闭环开始

按以下顺序实现，拿到结果后再扩展范围：

1. 从当前启动脚本追到真实 worker、模型构造器和 forward。复用已有入口、模型构建和权重加载逻辑；诊断副本中的补丁应有开关，并记录源码哈希。
2. 在**实际执行模型的进程**输出运行时绑定，确认改的是被加载的代码。
3. 对实际 decoder 层装观测 hook，先验证一次真实请求的输入、至少一个中间边界和最终端点；之后覆盖全部层。所有参与原 forward 的 rank 仍须正常参与计算。
4. 请求开始时启用 recorder，forward 完成后导出；同时记录命中的层／阶段／调用次数、rank、token 范围及 dtype。
5. 离线重建同一批逻辑 token，导出 skill 的 NPZ。已有符合约定的 cossim 时直接复用；否则运行 `compare_tensors.py` 计算并保存指标，这一步使用已采集张量，无需重新执行模型。
6. 保存同端无插桩与有插桩的对应端点、cosine／logprob 误差及重复波动，同时记录 graph、fusion、kernel 和执行时序变化。

层内多个中间结果按 [层内观测记录](operator-localization.md) 的字段导出，包含实际输入／输出采集点、cosine、增长幅度及其原始数据路径。已保存且对齐正确的指标直接引用。

实施代码建议分成 `capture_adapter.py`（记录器和探针）、`run_capture.py`（调用现有 forward）、`export_capture.py`（离线对齐）。这些是职责划分，不要求项目使用固定文件名。不要为了插桩重新实现 attention／MoE，也不要先把整个 RL 训练器和 optimizer 搬进最小探针。

没有历史 rollout token 时，按现有配置和输入数据新运行一次真实 rollout，保存其 token 并由 train 重放；数据来源标为新采集。用于插桩 smoke test 的任意合成 token 单独标为 synthetic，与真实 rollout 分开存档。

## 2. 确认实际绑定，避免“改了文件却没生效”

在框架完成 feature patch、模型构造和相关初始化之后检查。每个不同角色的 worker 至少留一份；主进程打印不能证明 Ray／spawn 子进程加载了相同对象。

```python
import hashlib
import inspect
import sys
from pathlib import Path

def describe_callable(fn):
    bound = getattr(fn, "__func__", fn)
    resolved = inspect.unwrap(bound)
    record = {"python": sys.executable, "module": getattr(bound, "__module__", None),
              "qualname": getattr(bound, "__qualname__", type(bound).__name__)}
    for name, target in (("bound", bound), ("unwrapped", resolved)):
        code = getattr(target, "__code__", None)
        record[name + "_code_file"] = getattr(code, "co_filename", None)
        record[name + "_code_line"] = getattr(code, "co_firstlineno", None)
        try:
            source = inspect.getsourcefile(target)
        except (TypeError, OSError):
            source = None
        record[name + "_source"] = source
        if source and Path(source).is_file():
            record[name + "_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return record
```

同时保存 `type(model)`、`type(layer)`、`type(layer.mlp)`、实际 `forward` 和关键被调用函数的描述，以及包的 `__file__`、有效配置。`inspect.unwrap` 不保证穿透所有装饰器：必要时检查 `__globals__` 中实际查找的函数、闭包和短调用 trace。若调用方使用 `from package import op`，只替换 `package.op` 可能不会影响已绑定的调用方；修改实际查找位置并检查命中次数。

安装点选择：

| 执行形式 | 安装位置 |
|---|---|
| Megatron／MindSpeed model provider | 原 feature patch／配置初始化之后、真实模型构造后，访问实际 `model.decoder.layers`；先核对 DDP／其他 wrapper 解包方式 |
| vLLM worker 内模型 | 在 worker 加载模型的插件／初始化入口装 hook；或使用已验证支持的注册模型子类，在其构造函数里安装 |
| Ray／spawn 多进程 | 通过现有 worker 初始化、模块导入或插件机制部署；记录每个目标 worker 的源码指纹和安装日志 |
| torch.compile／graph 路径 | 先检查探针是否执行以及 graph 是否变化；需要 eager 对照时单独标记，不能默认替代原路径 |

当目标 vLLM 版本支持模型懒加载注册时，可采用以下形式。`CapturedModelForCausalLM` 和 `capture_adapter` 是示例名称，需由适配代码提供实际实现：

```python
def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "CapturedModelForCausalLM",
        "capture_adapter:CapturedModelForCausalLM",
    )
```

对应模型副本的 architecture、worker 可导入的 `capture_adapter`、返回的实际类必须一致。采集子类继承目标版本的原生模型，装 hook 后调用 `super().forward(...)`；不能只注册名称而遗漏子进程可见的代码。某版本不支持同样的 `model_cls` 扩展时，改用该版本已存在的 worker 模型加载入口。

## 3. 被动快照：复制数据，保持原返回值

下面的 recorder 接收**已经由适配层解释成 token 行**的张量。可保留 `[N,H]` 或 `[N,C,H]`，不在采集时合并 token 维。`row_keys` 是本 rank 这些行的逻辑位置，不是局部行号冒充全局 token。

```python
from pathlib import Path
import torch

class Recorder:
    def __init__(self):
        self.scope = None
        self.buffers = {}

    def begin(self, scope):
        assert self.scope is None
        self.scope = dict(scope)
        self.buffers = {}

    def put(self, stage, rows, row_keys, metadata):
        if self.scope is None:
            return
        assert stage not in self.buffers, (stage, "duplicate capture")
        assert isinstance(rows, torch.Tensor) and rows.ndim >= 2
        assert len(row_keys) == rows.shape[0]
        self.buffers[stage] = {
            "hidden": rows.detach().clone(),
            "token_keys": list(row_keys),
            "source_shape": list(rows.shape), "dtype": str(rows.dtype),
            "layout": dict(metadata),
        }

    def finish(self, filename):
        assert self.scope is not None
        # Called after the captured forward, with the framework's stream
        # dependencies satisfied. No per-boundary CPU transfer or collective.
        stages = {name: {**item, "hidden": item["hidden"].cpu()}
                  for name, item in self.buffers.items()}
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"scope": self.scope, "stages": stages}, tmp)
        tmp.replace(path)
        self.scope, self.buffers = None, {}

    def abort(self):
        self.scope, self.buffers = None, {}
```

关键要求：

- `detach()` 仍与原 tensor 共享存储；必须 `clone()`，或用产生独立快照的 `index_select()`。否则下游 in-place 更新会改掉“之前”的观测值。
- 不在每个 hook 中新增 `.cpu()`、`.item()`、打印整 tensor 或 collective。先保留设备快照，再在明确的完成边界转 CPU；多 stream 需使用框架正确的依赖／同步。
- 保留原 dtype，计算指标时再升精度。不对记录值做归一化，也不修改原 tensor。
- 内存不足时先缩小一次保留的层／内部阶段范围；可分多次固定轨迹采集。只保存部分 token 时必须报告覆盖范围，子集均值不能标成所有有效 token 均值。
- 这个简化 recorder 一次只处理一个 capture scope，同一 stage 只能出现一次。动态 batching／多次 decode 要把 request、step、microbatch、调用次数纳入键；不要静默保留第一次或覆盖最后一次。
- scope 至少含运行 ID、请求／序列 ID、phase、forward 序号、rank 和 token 映射来源；文件名也含 rank／调用 ID，避免多进程覆盖。

在实际 worker 的 forward 边界连接记录器；`call_original` 调用原有 forward，`check_coverage` 核对该 worker／请求预期执行的阶段。不要只在外层 `LLM.generate()` 主进程设置 recorder，而误以为子进程会共享它。

```python
def capture_one_forward(call_original, recorder, scope, output_path, check_coverage):
    recorder.begin(scope)
    try:
        result = call_original()
        check_coverage(recorder.buffers)
        recorder.finish(output_path)
        return result
    finally:
        recorder.abort()
```

hook handles 可在模型生命周期安装一次，由 scope 控制是否记录；模型销毁或诊断结束时逐一 `handle.remove()`。一次 forward 失败时清空未完成快照，不能把它当作完整采集。

## 4. 逐层与层内 hook：先找语义对应点

`to_rows(stage, value, scope)` 是当前项目的布局适配函数，返回 `(rows, row_keys, metadata)`。`select_value(output)` 必须根据实际返回契约选择数据；不能无条件将所有 tuple 的第 0 项都当作层完整输出。独立 residual、bias、cache 或多流状态按语义另存。

```python
def install_output_hook(module, stage, recorder, to_rows, select_value):
    def observe(_module, _args, output):
        if recorder.scope is not None:
            value = select_value(output)
            rows, keys, meta = to_rows(stage, value, recorder.scope)
            recorder.put(stage, rows, keys, meta)
        return None  # nn.Module hook returns None to preserve original output.
    return module.register_forward_hook(observe)

def install_input_hook(module, stage, recorder, to_rows, select_input):
    def observe(_module, args, kwargs):
        if recorder.scope is not None:
            value = select_input(args, kwargs)
            rows, keys, meta = to_rows(stage, value, recorder.scope)
            recorder.put(stage, rows, keys, meta)
        return None
    return module.register_forward_pre_hook(observe, with_kwargs=True)
```

对实际层对象逐一调用安装函数并保留 handles。每层使用独立 stage／全局层号，避免 Python 循环闭包把所有层记到同一个 index。PP 的本地 index 不一定等于全局层号；若一端从 1 编号、另一端从 0 编号，应显式保存转换关系。

以下是常见属性名的语义映射示例，实际名称和边界以运行时源码为准：

| 语义 stage | 训练端示例对象 | 推理端示例对象 |
|---|---|---|
| attention 前 norm | `input_layernorm` | `input_layernorm` |
| attention 输出 | `self_attention` | `self_attn` |
| FFN 前 norm | `pre_mlp_layernorm` | `post_attention_layernorm` |
| MoE 输出 | `mlp` | `mlp` |

按这些名称读取当前对象并检查源码，名称不匹配就沿实际 forward 寻找等价边界。对多流或多分支结构，另存残差混合、权重等后续计算依赖，不能只取一个 hidden 就声称边界状态完整。

没有独立 activation module 时，优先捕获其消费者的输入。例如可用 expert MLP 的 FC2 pre-hook 观察真正送入 down projection 的激活：

```python
# Illustrative expert attributes; resolve the actual runtime bindings first.
fc1_handle = install_output_hook(
    expert.linear_fc1, "expert.fc1", recorder, to_rows, select_fc1_output)
activation_handle = install_input_hook(
    expert.linear_fc2, "expert.activation_to_fc2", recorder, to_rows,
    lambda args, kwargs: kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])
```

此值可能已经乘过路由权重或经过 cast，必须标注为“FC2 实际输入”，不能未经核对当作纯激活函数输出。shared 与 routed 分支分别布置探针，多个 expert 的键含全局 expert ID。

## 5. 非 module 方法、函数和融合区间

方法不走 `nn.Module.__call__` 时，使用只转发原方法的包装器。例如残差混合方法、router gating 和 fused MoE 的分支返回值：

```python
from types import MethodType

def wrap_method(obj, name, before=None, after=None):
    original = getattr(obj, name)  # Bound method: do not pass self twice.
    def wrapped(self, *args, **kwargs):
        if before is not None:
            before(args, kwargs)
        result = original(*args, **kwargs)
        if after is not None:
            after(result, args, kwargs)
        return result
    setattr(obj, name, MethodType(wrapped, obj))
    return original
```

回调只记录，返回值保持原对象；结束时恢复原方法。对一次 forward 内两次调用的方法，优先按显式 stage 参数区分；无参数时可按源码确认的调用顺序计数，但必须每个请求重置并断言次数。顺序不符时记录 stage 映射错误，不把额外调用标成 FFN。

独立 Python 函数可在其真实调用位置安装同样的透明包装。若需观察函数内部局部变量：优先在诊断源码副本的准确语义边界插入 `recorder.put(...)`；或只对已解析函数的 `__code__` 做短时 `sys.settrace`。例如在原生 `output += shared_expert_output` 前记录两分支，避免原地合并后丢失各自值。

局部变量探针必须根据当前源码定位唯一锚点，记录文件哈希，限定目标请求，在 `finally` 中恢复原 trace。Python line event 在该行执行前触发：要抓算子输出应放在它执行完成后的下一边界。compiled／TorchScript／C++ 内部不保证可见，Python trace 无记录不代表该算子没执行。

没有可观测内部边界的 fused kernel 记录可见输入／输出，内部点标为 unavailable。受控变体、拆分实现和手写参考各自保存输入／输出、代码及路径信息，与原 fused 数据分开存档。

## 6. rank／token／特征布局：离线对齐后再算 cosine

为避免在每个观测点加入通信，各 rank 保存本地快照和映射，在离线脚本重建。

| 实际布局 | 处理方法 |
|---|---|
| 已验证的连续 sequence shard | 全局位置 = 该 shard 的真实 offset + 本地行号；合并 token 行并排除 padding |
| TP 完整副本 | 同一逻辑 replica 只计一次；必要时核对其他副本，不直接拼接产生重复 token |
| TP feature shard | 按真实特征顺序拼成完整 token 向量，不能平均各 shard 的 cosine |
| 部分和／未归约输出 | 先标明 reduction 状态；优先采原生归约后的输出。离线高精度求和只是参考，不能冒充原生归约结果 |
| EP dispatch 后 | 用 `(request, token_position, global_expert_id, occurrence)` 对齐，保留源 DP／replica 信息 |
| packed／动态 batch／context parallel | 从真实 metadata 获取 token 映射；不能默认 `rank*local_length` |

若当前 train 布局为 `[S_local,1,C,H]`，可在验证后映射为 `value[:,0]`；对应 rollout 布局可能为 `[N,C,H]` 或连续 sequence shard。只有验证了 B=1、完整 hidden 特征、连续均分序列时，才可使用：

```python
# Special-case adapter, valid ONLY for the verified contiguous SP layout.
assert value.ndim == 4 and value.shape[1] == 1
rows = value[:, 0]
assert rows.shape[0] * tp_size == padded_length
offset = tp_rank * rows.shape[0]
keep = [p for p in valid_positions if offset <= p < offset + rows.shape[0]]
indices = torch.tensor([p - offset for p in keep], device=rows.device, dtype=torch.long)
snapshot = rows.detach().index_select(0, indices)
keys = [f"{request_id}:position-{p}" for p in keep]
```

B>1 必须结合样本维和 packing 映射拆开；不能直接 `squeeze()` 后猜 token 轴。rollout 也不能仅凭第一维长度碰巧相同就判成完整副本，须结合实际通信模式／scheduler metadata。

还需检查特征顺序：gate/up 的每个 shard 为 `[g_rank,u_rank]` 时，直接按 rank 拼接会得到交错顺序。只有检查了权重与切分契约后，使用：

```python
halves = [part.chunk(2, dim=-1) for part in parts_in_tp_rank_order]
gate = torch.cat([pair[0] for pair in halves], dim=-1)
up = torch.cat([pair[1] for pair in halves], dim=-1)
logical_gate_up = torch.cat([gate, up], dim=-1)
```

EP 重排优先使用原 dispatch／unpermutation 索引。如果缺少索引而尝试用输入行指纹筛选候选，再逐元素验证整行来恢复 token，这只在值未改变、候选行能唯一确认时成立。重复输入、cast 或缩放会使该办法失效，不能用最近邻或任意“第一个匹配”补齐。一个 token 经过多个 expert，要分别保留 expert 维及路由集合差异；共同 expert 子集与全 MoE 输出分别存档，记录覆盖范围。

离线合并后断言：key 无遗漏、无意外重复、集合及顺序与 manifest 相同；原 token ID、mask、特征布局都已核对。再把每个 token 的特征展平为一行 `[N,D]`，BF16 无损提升到 FP32，输出 `values`、Unicode `token_keys`、bool `valid_mask`。需要补算指标时使用 skill 随附脚本；已有正确指标可复用，但不套用其他实验的样本数或阈值。

## 7. 共同输入注入：在目标边界接入保存值

先保存边界的完整实际输入与状态，不只保存用于统计的少数 token。层重放需要保留上下文、padding 和原生执行条件。若两端使用不同前缀长度，先证明 token 对齐再做映射；不能通过无依据的切片消除位置差异。

例如可把 train 的 norm 后输出注入 rollout 的 MoE 入口；后续仍运行推理端原生 router、dispatch 和 experts。核心 pre-hook 模式如下。`get_common()` 返回当前 scope 预先加载、已验证布局／dtype／device 的输入，不在 hook 中读磁盘或静默转换：

```python
def install_input_replay(module, get_common, enabled, input_name, input_index):
    def replace(_module, args, kwargs):
        if not enabled():
            return None
        old = kwargs[input_name] if input_name in kwargs else args[input_index]
        common = get_common()
        assert common.shape == old.shape
        assert common.dtype == old.dtype and common.device == old.device
        # Protect the saved common input from downstream in-place operations.
        injected = common.detach().clone()
        if input_name in kwargs:
            return args, {**kwargs, input_name: injected}
        modified = list(args)
        modified[input_index] = injected
        return tuple(modified), kwargs
    return module.register_forward_pre_hook(replace, with_kwargs=True)
```

由当前 `forward` 签名确定 `input_index`：某端 hidden 可能在 `args[0]`，另一端 decoder 的 `args[0]` 却可能是 positions。不要把整层 hook 的参数位置照搬到 `mlp`。

注入 hook 安装在输入观测 hook 之前（或在注入处同步记录），确认保存的是**实际收到的注入值**。先完成 T(xT)、I(xI) 自重放控制，再跑交叉输入并回读验证入口值；相同 dtype 且未做数值变换时可检查保存／重载后的字节一致性。

保留 router replay 原开关。对包含 router 的 MoE 区间，路由结果是内部输出，不随 hidden 一起强制替换。该代码模式只注入 hidden；attention 或其他有状态区间的历史输入需用对应适配器另行注入／对齐，并保存完整字段清单。

## 8. 资源不足时的分段 runner

可用“加载若干原始层 → 执行完整上下文 → 保存边界状态 → 退出进程 → 下一段”降低权重驻留量。仅在原始路径无法按现有资源运行且需要此对照时采用；先记录它改变的并行、batch、cache 或初始化条件。

- 使用原始 layer 实现、配置和原全局层号；权重映射不能因截取层段而错位。
- 保存后续段真正需要的状态。一般模型可能需要 residual、cache、position 和其他流；某个边界可仅传 hidden，不代表其他模型也如此。
- 恢复到下一段原生 layer 输入，不能把上一段临时 norm／LM head 的输出当作 decoder hidden。
- 如引擎要求执行临时 head 才能结束请求，可完成该请求，但其分段 logits 不作为原模型的端到端 logprob。
- 比较相同条件的连续短段与保存／重载短段，在边界和后续端点做同端控制；扩展层数时保留对应控制结果。分段与原始运行分别记录执行条件和指标。
- 注意进程重启会重置懒初始化、全局缓存和首次调用状态；这也是分段对照要验证的条件，不能只看 tensor 保存无损。

## 9. 卡住时按证据换实施方式

| 技术症状 | 代码排查方法 |
|---|---|
| 文件已改但 hook 次数为 0 | 检查 worker 解释器、`__file__`、实际模型类、是否经过 `__call__`、注册是否在子进程生效；查看请求 gating 是否把真请求排除 |
| 只有 warmup／profile 数据 | 使用真实 request／step metadata 标识目标执行；完整 token 列表哈希只适合已验证的单请求 full-prefill，不能当通用 decode 标识 |
| PP 后段没有 input_ids | 通过 runner 的 request／microbatch 映射传递身份；共享请求标记文件仅适合严格串行单请求，不可直接用于并发服务 |
| 找不到 activation module | 抓下一算子的输入，例如 down projection pre-hook；检查此处是否已缩放／cast |
| fusion 内变量看不见 | 先定界原区间，再用准确源代码探针、短时 Python trace 或受控变体；不用自写替代函数冒充观测 |
| 相同 stage 多次出现 | 明确调用序号、分支、microbatch、decode step、recompute；不要静默覆盖 |
| rank 合并后多／少 token | 查 SP／TP／EP／DP／PP 语义、padding、副本和 dispatch 索引，不通过截断或取 mask 交集消除错误 |
| 同输入但入口值不相同 | 查 hook 顺序、参数位置、dtype／布局变换、in-place 覆盖、状态遗漏；修复后回读实际入口值与完整状态 |
| 插桩后端点改变 | 缩小探针、去掉逐点同步／额外通信，检查编译路径和时序；保留差异记录，不能用插桩输出覆盖原始基线 |
| 保存所有层 OOM | 固定轨迹分批采集层／阶段；必要时用已验证分段 runner，明确覆盖与执行条件 |

hook／方法包装要在 `finally` 清理或恢复；失败快照不要写成完成证据。采集验收至少检查：worker 安装日志、预期层覆盖、stage 顺序／次数、token key 覆盖、dtype／布局、输入来源、自重放控制和插桩端点控制。
