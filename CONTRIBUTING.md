# 贡献说明

优先贡献可复现的指标问题、框架适配和诊断案例。描述改动解决的具体问题、输入输出契约、已验证版本和未覆盖条件。

数值改动需要使用能暴露差异的样例验证：padding、零范数、非有限值、token 重排、幅值变化和 chunk 边界均应保持语义。运行下文的两组 unittest 与发布检查。不要把通过指标单测表述为通过真实引擎验证。

框架适配应保留原 forward 路径、token／rank 映射、完整状态、自重放控制和插桩前后端点。若改变 dtype、并行、batch、fusion 或 cache，请将其作为独立实验变量记录。公开案例应包含可运行命令、环境、原始结果和结论边界。

添加要分发的文件时更新 `release-files.txt`。样本和模型须有可公开使用的来源；不要提交凭据、私有 prompt、权重、trace、系统元数据或本地路径。第三方代码需保留其来源与要求的声明。授权状态以 PROVENANCE.md 为准。

## 测试

在仓库根目录的 Python 虚拟环境中运行：

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s skills/rl-train-infer-diagnosis/scripts -p 'test_*.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/build_release.py --check
```

离线检查包括指标／CLI、合成流程和发布测试，并检查 ZIP 解包和 skill 单独安装后的运行。本地环境为 macOS、Python 3.9.6、NumPy 1.26.4。GitHub Actions 配置覆盖 Python 3.10 和 3.12，远程结果以实际 CI 为准。这些离线检查与 skill 已有的实际训推诊断验证分别记录。

## 打包

审查草稿：`python tools/build_release.py --draft`。

生成发布包：`python tools/build_release.py`。该命令检查公开发布授权与文件清单；若声明了许可证，还会检查许可元数据与许可文件是否一致。

打包仅包含 `release-files.txt` 中明确列出的文件，不递归打包工作区。生成物排除虚拟环境、系统元数据、原始 trace 和本地审查材料。目标 ZIP 已存在时会停止，可使用 `--destination` 指定新的输出目录。

## 分享实验结果

提交版本、输入布局、复现命令和实际指标，明确适用条件及未执行的实验。CLI 默认省略输入文件路径；如需本地追溯，可使用 `--path-mode absolute`。省略路径不会自动脱敏 prompt、token key、配置、命令或张量内容。

## 安装维护

安装 skill 时为执行工具的 Python 环境安装其 `requirements.txt`。升级前检查原安装目录中的本地修改；卸载时删除此前安装的 `rl-train-infer-diagnosis` 目录。实验产物保存在任务工作区。Codex 的安装与发现规则见 [官方说明](https://learn.chatgpt.com/docs/build-skills)。
