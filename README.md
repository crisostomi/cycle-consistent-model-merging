# Cycle Consistent Model Merging

<p align="center">
    <a href="https://github.com/crisostomi/cycle-consistent-model-merging/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/crisostomi/cycle-consistent-model-merging/Test%20Suite/main?label=main%20checks></a>
    <a href="https://crisostomi.github.io/cycle-consistent-model-merging"><img alt="Docs" src=https://img.shields.io/github/deployments/crisostomi/cycle-consistent-model-merging/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.3-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

![Teaser](figures/teaser.png)

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

All the scripts can be found under `src/scripts/`. Each script has a corresponding configuration file in `conf/matching` where you can change stuff as dataset and model to use. You can train models using `train.py` with a dataset and model of your choice.

### Matching two models

1. get the permutations to align the two models (identified by their seed in the config) by running `match_two_models.py`. The config is `conf/matching.yaml` (see inside the config to see the subconfigs).
2. evaluate the interpolation of the models using `evaluate_matched_models.py` and the same config used for the previous step. Be sure to have `matching.yaml` as config in the script itself.

To change the matching technique, you have to change the `matcher` in `conf/matching/match_two_models.yaml`. Each matcher has its own config file in `conf/matching/matcher/`.

To run all the pairs of models with different seeds, run `shell_scripts/run_all_seeds.sh`.

### Matching multiple models

1. get the permutations to align the models (identified by their seed in the config) by running `match_n_models.py`. The config is `conf/matching_n_models.yaml` (see inside the config to see the subconfigs).
2. evaluate the interpolation of the models using `evaluate_matched_models.py` and the same config used for the previous step.  Be sure to have `matching_n_models.yaml` as config in the script itself.

To change the matching technique, you have to change the `matcher` in `conf/matching/match_n_models.yaml`. Each matcher has its own config file in `conf/matching/matcher/`.

### Merging models

1. get the merged model by running `merge_n_models.py`.
2. evaluate the merged model using `evaluate_merged_model.py` and the same config used for the previous step.

To change the merging technique, you have to change the `merger` in `conf/matching/merge_n_models.yaml`. Each merger has its own config file in `conf/matching/merger/`.
