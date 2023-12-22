# Cycle Consistent Model Merging

<p align="center">
    <a href="https://github.com/crisostomi/cycle-consistent-model-merging/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/crisostomi/cycle-consistent-model-merging/Test%20Suite/main?label=main%20checks></a>
    <a href="https://crisostomi.github.io/cycle-consistent-model-merging"><img alt="Docs" src=https://img.shields.io/github/deployments/crisostomi/cycle-consistent-model-merging/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.3-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Merging models in a cycle-consistent fashion.

## Development installation

Setup the development environment:

```bash
git clone git@github.com:crisostomi/cycle-consistent-model-merging.git
cd cycle-consistent-model-merging
conda env create -f env.yaml
conda activate ccmm
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```

### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```

## Usage

### Git-Rebasin

All the scripts can be found under `src/scripts/`. Each script has a corresponding configuration file in `conf/matching` where you can change stuff as dataset and model to use.

1. train your models using `train.py`, making sure to change the random seed so to have two different modes
2. get the permutations to align the two models (identified by their seed in the config) using `match_and_sync.py` with config `git_rebasin``
3. evaluate the interpolation of the models using `evaluate_matched_models.py` and the same config used for the previous step.
4. results are found in `results/${dataset}/match_and_sync/None`
