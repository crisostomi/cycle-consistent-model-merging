import copy
import json
import logging
import os
from typing import Dict

import hydra
import omegaconf
import pandas as pd
import torch  # noqa
import torchvision.transforms as transforms
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from scripts.evaluate_matched_models import evaluate_pair_of_models

import ccmm  # noqa
from ccmm.matching.matcher import GitRebasinMatcher
from ccmm.matching.utils import (
    apply_permutation_to_statedict,
    get_all_symbols_combinations,
    plot_permutation_history_animation,
    restore_original_weights,
)
from ccmm.models.resnet50 import PretrainedResNet50
from ccmm.pl_modules.pl_module import MyLightningModule
from ccmm.utils.utils import get_model, load_model_from_artifact, map_model_seed_to_symbol, save_factored_permutations

pylogger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        path = self.dataframe.loc[index, "path"]
        label = self.dataframe.loc[index, "label"]
        image = Image.open(path).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        image = transform(image)
        return image, label


def load_imagenet(data_path):
    paths = []
    labels = []
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename[-4:] == "JPEG":
                paths += [(os.path.join(dirname, filename))]
                label = dirname.split("/")[-1]
                labels += [label]

    class_names = sorted(set(labels))
    N = list(range(len(class_names)))
    normal_mapping = dict(zip(class_names, N))

    df = pd.DataFrame(columns=["path", "label"])
    df["path"], df["label"] = paths, labels
    df["label"] = df["label"].map(normal_mapping)

    return CustomDataset(df)


seed_to_timm_model = {
    0: "a1",
    1: "a2",
    2: "a3",
    3: "am",
    4: "b1k",
    5: "b2k",
    6: "c1",
    7: "c2",
    8: "a1h",
    9: "bt",
    10: "ra",
    11: "ram",
    12: "gluon",
    13: "tv2",
}


def run(cfg: DictConfig) -> str:
    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    data_path = PROJECT_ROOT / "data/imagenet-mini"

    train_dataset = load_imagenet(data_path)
    train_loader = DataLoader(train_dataset, batch_size=100, num_workers=8)

    # {a1: 1, a2: 2, b1k: 3, ..}
    symbols_to_seed: Dict[int, str] = {seed_to_timm_model[seed]: seed for seed in cfg.model_seeds}

    run = wandb.init(project=core_cfg.core.project_name, entity=core_cfg.core.entity, job_type="matching")

    # {a: model_a, b: model_b, c: model_c, ..}
    models: Dict[str, LightningModule] = {
        seed_to_timm_model[seed]: MyLightningModule(
            model=PretrainedResNet50(weights=seed_to_timm_model[seed], num_classes=1000), num_classes=1000
        )
        for seed in cfg.model_seeds
    }

    model_orig_weights = {symbol: copy.deepcopy(get_model(model).state_dict()) for symbol, model in models.items()}

    ref_model = list(models.values())[0]

    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation_spec(ref_model)

    # always permute the model having larger character order, i.e. c -> b, b -> a and so on ...
    symbols = set(symbols_to_seed.keys())

    # (a, b), (a, c), (b, c), ...
    all_combinations = get_all_symbols_combinations(symbols)
    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc
    # TODO: understand if it's important for the combinations to be all possible ones or just the ones that are unique
    canonical_combinations = [(source, target) for (source, target) in all_combinations if source < target]  # NOQA

    all_accs = {symbol: {} for symbol in symbols}
    for fixed_symbol, permutee_symbol in tqdm(canonical_combinations):

        fixed_model, permutee_model = models[fixed_symbol].cpu(), models[permutee_symbol].cpu()

        # dicts for permutations and permuted params, D[a][b] refers to the permutation/params to map b -> a
        gitrebasin_permutations = {
            symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols
        }

        matcher = GitRebasinMatcher(name="git_rebasin", permutation_spec=permutation_spec)

        restore_original_weights(models, model_orig_weights)

        gitrebasin_permutations[fixed_symbol][permutee_symbol], perm_history = matcher(
            fixed=get_model(fixed_model), permutee=get_model(permutee_model)
        )

        restore_original_weights(models, model_orig_weights)

        updated_params = {fixed_symbol: {permutee_symbol: None}}

        pylogger.info(f"Permuting model {permutee_symbol} into {fixed_symbol}.")

        # perms[a, b] maps b -> a
        updated_params[fixed_symbol][permutee_symbol] = apply_permutation_to_statedict(
            permutation_spec,
            gitrebasin_permutations[fixed_symbol][permutee_symbol],
            get_model(models[permutee_symbol].cpu()).state_dict(),
        )
        restore_original_weights(models, model_orig_weights)

        lambdas = [0.5]  # np.linspace(0, 1, num=4)

        repair_models = {fixed_symbol: models[fixed_symbol], permutee_symbol: models[permutee_symbol]}

        results = evaluate_pair_of_models(
            repair_models,
            fixed_symbol,
            permutee_symbol,
            updated_params,
            train_loader,
            test_loader=None,
            lambdas=lambdas,
            cfg=core_cfg,
        )

        merged_acc = results["train_acc"]
        all_accs[fixed_symbol][permutee_symbol] = merged_acc

    print(all_accs)

    with open("results.json", "w+") as f:
        json.dump(all_accs, f)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching_n_models", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
