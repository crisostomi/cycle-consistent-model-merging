import os

import hydra
from datasets import load_dataset
from hydra import compose, initialize

from nn_core.common import PROJECT_ROOT

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path=str(f"{PROJECT_ROOT}/conf/dataset/"), job_name="matching_n_models")

cfg = compose(config_name="tiny_imagenet", overrides=[])

ref = "Maysee/tiny-imagenet"

train_split = "train"
test_split = "valid"

label_key = "label"
image_key = "image"

train_hf_dataset = load_dataset(
    ref,
    split=train_split,
    use_auth_token=True,
)

test_hf_dataset = load_dataset(
    ref,
    split=test_split,
    use_auth_token=True,
)


dataset_name = "tiny_imagenet"
splits = ["train", "test"]

base_dir = "../../data/" + dataset_name

datasets = {"train": train_hf_dataset, "test": test_hf_dataset}

for split in splits:
    classes = datasets[split].features["label"].names

    for class_name in classes:
        os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

    for i, example in enumerate(datasets[split]):
        image = example["image"]
        label = example["label"]
        class_name = classes[label]

        image_path = os.path.join(base_dir, split, class_name, f"{i}.jpg")
        image.save(image_path)
