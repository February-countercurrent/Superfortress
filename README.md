# Superfortress

Bomberman reinforcement learning for Machine Learning Essentials, Summer Semester 2026.

This repository presents two trained agents directly. Complete training, evaluation, diagnostic and historical model source code is included in [`research/source.zip`](research/source.zip), with a one-command extraction tool. The report manuscript is not included.

| Agent | Method | Role |
| --- | --- | --- |
| `Superfortress` | Seven-block Q-learning, D4 sharing and explicit safety | Exact submitted agent, 1,150-episode training lineage |
| `tree_teacher` | 28-feature ExtraTrees fitted Q iteration | Retained later research model trained partly on demonstrations |

The `tree_teacher` entry point pairs the unchanged `tree_guard` inference source with the retained demonstration checkpoint. It is a packaging alias, not a newly trained model. The older random-data control has not been substituted for it.

## Run

Use Python 3.10. The recorded tree environment used scikit-learn 1.2.1; other versions may not load its serialized weights correctly.

```sh
python -m pip install -r requirements.txt
python main.py play --no-gui --agents Superfortress rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 10 --seed 20260929
python main.py play --no-gui --agents tree_teacher rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 10 --seed 20260929
```

Remove `--no-gui` to watch. Clear any old `FQI_MODEL` environment override before running `tree_teacher`; otherwise its original callback honors that override. These commands evaluate frozen weights. Training code is in the research archive. Docker execution was not tested locally.

## Recorded results

| Independent test | Strict wins | Mean score |
| --- | ---: | ---: |
| Submitted Superfortress | 65/120 (54.2%) | 6.767 |
| Demonstration tree | 292/600 (48.7%) | 6.292 |
| Random-data tree control, same 600 tree-test maps | 266/600 (44.3%) | 5.913 |

A strict win is a uniquely highest final game score. The tree comparison has a 95% bootstrap win-difference interval of approximately [-0.84, +9.67] percentage points; a stable advantage remains unconfirmed. The submission and tree tests used different maps and are not a direct comparison. These are local benchmarks, not tournament results.

Original summaries: [`evidence/submission.json`](evidence/submission.json) and [`evidence/tree_confirmation.json`](evidence/tree_confirmation.json). Other report evidence is inside the research archive at its original relative paths.

## Complete source and historical models

```sh
python restore_research.py
```

This expands the full research source and selected experiment records into `research/workspace/`, restoring the original layout and copying the two provided checkpoints to their original locations. All developed model implementations, including early tabular agents, DQN and unsuccessful variants, remain available there. Exact duplicate historical source snapshots are also restored.

Historical trained weights and raw trajectory datasets are omitted from this small upload. The two main agents run immediately; rerunning older experiments may require retraining or recollecting data. Frozen-result reproduction is therefore limited to the artifacts actually included. See [`research/README.md`](research/README.md) and [`METHODS_INVENTORY.md`](METHODS_INVENTORY.md) for entry points and development history. Restoring the archive does not overwrite modified files.

## Contents

- `agent_code/`: two main agents and the supplied framework opponents.
- `evidence/`: the main recorded result summaries.
- `research/`: complete research source archive, integrity index and extraction instructions.
- `release/final-project-agent-code.zip`: unchanged single-agent competition submission.
- `verify_publication.py`: verifies files, source archive and protected model/package hashes.

The game framework and original opponents come from [ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl); source revision and attribution are in [`UPSTREAM.md`](UPSTREAM.md). Development and documentation received substantial AI assistance. No university report or course handout is included, and no new license is assigned to upstream code or assets.
