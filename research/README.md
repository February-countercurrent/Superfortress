# Research source archive

Run `python restore_research.py` from the repository root. It verifies archive hashes and extracts to `research/workspace/`, which is ignored by Git. Open that directory to inspect all original model, training, evaluation, analysis, test and verification code, together with selected experiment evidence and development notes. The public repository presents the two main inference agents directly; the archive keeps the historical source available without hundreds of loose files in the main folder.

Important original paths after extraction:

| Path | Purpose |
| --- | --- |
| `agent_code/q_linear/`, `q_dynamic/`, `q_guard_train/` | Block-Q learning and later training variants |
| `agent_code/tree_fqi/`, `tree_tactical/`, `tree_guard/` | Tree-based FQI and inference variants |
| `agent_code/dqn_agent/` | DQN pilot architecture and learning code |
| `fit_tree_fqi.py`, `fit_tree_tactical.py` | Offline tree fitting |
| `experiment_tree_teacher.py` | Demonstration-data collection and experiment orchestration |
| `experiment_*.py`, `run_*.py` | Experiment and evaluation entry points |
| `test_*.py`, `verify_*.py`, `analyze_*.py` | Tests, audits and analysis |
| `experiments/` | Selected recorded results/configurations used in the report |

The submitted Q weights and retained demonstration-tree weights are copied from the main agents into their original research paths. In particular, the teacher weights go to `experiments/tree_teacher_v1/fit/model.pkl`; they are **not** assigned to the historical random-data control at `agent_code/tree_guard/model.pkl`.

Older checkpoints, DQN weights, raw transition datasets and detailed replay traces are not included in this small public export. Original experiment records may reference these omitted files or local machine paths. Some historical runners protect fixed output directories or check original checkpoint hashes. Inspect and configure those runners for a new experiment; do not expect every archived command to replay immediately without its original inputs. Raw trajectories must be recollected for fresh training.

The archived `requirements-research.txt` records NumPy 1.23.5, scikit-learn 1.2.1, pygame 2.5.0 and matplotlib 3.7.0. The DQN pilot additionally used PyTorch 1.12.1. The main inference requirements are separately listed at the repository root.

The code archive preserves every distinct project Python implementation from the preceding publication export. `research/index.json` records each archive member's path, size and SHA-256. This archive contains project development records; the course report manuscript remains outside the repository.
