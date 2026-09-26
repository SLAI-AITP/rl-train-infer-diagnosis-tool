# RL Train–Infer Diagnosis

An Agent Skill for locating numerical differences between the forward passes of RL rollout and training replay using the same weight snapshot and token trajectory.

Through layer-by-layer comparisons, bidirectional replay with identical inputs, and interventions that change one factor at a time, it produces diagnostic reports covering **where divergence occurs, replay evidence, intervention effects, and unresolved questions**. This skill has been validated in real training–inference diagnosis workflows. For other projects, adapt trace capture and replay to the model and framework versions in use.

## Installation

The following example uses Codex. From the repository root, use Python 3.9–3.12 to create a separate environment for the metrics tool and copy the skill to your Codex user directory. The copy step fails if the destination directory already exists.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python - <<'PYTHON'
from pathlib import Path
from shutil import copytree
import sys

copytree('skills/rl-train-infer-diagnosis',
         Path.home() / '.agents/skills/rl-train-infer-diagnosis')
print('Python interpreter for offline metrics:', Path(sys.executable).absolute())
PYTHON
```

Save the printed interpreter path and include it in your request to Codex. This environment runs the NumPy metrics tool; model trace capture and replay use your project's own runtime environment.

## Usage Example

Replace the placeholders below with your project entry points, input sources, and interpreter path, then send the request in Codex:

```text
Use $rl-train-infer-diagnosis to locate forward-pass differences
between rollout and training replay for the same batch of tokens.

Training entry point: <code path or launch command>
Inference entry point: <code path or launch command>
Weights and samples: <snapshot ID, sample source, or existing trace path>
Python interpreter for offline metrics: <absolute path printed during installation>

Report the evidence supporting the diagnosis, intervention results,
and any unresolved questions.
```

If no traces are available, new ones can be collected with access to the code, model weights, and compute resources.

## License

This project is licensed under the [MIT License](LICENSE).
