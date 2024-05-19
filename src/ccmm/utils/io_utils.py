from collections import namedtuple
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from nn_core.common import PROJECT_ROOT

from ccmm.data.my_dataset_dict import MyDatasetDict
from ccmm.utils.utils import convert_to_rgb


def load_hf_dataset(ref, train_split, test_split, use_cached):
    DatasetParams = namedtuple("DatasetParams", ["name", "fine_grained", "train_split", "test_split", "hf_key"])
    dataset_params: DatasetParams = DatasetParams(
        ref,
        None,
        train_split,
        test_split,
        (ref,),
    )
    DATASET_KEY = "_".join(
        map(
            str,
            [v for k, v in dataset_params._asdict().items() if k != "hf_key" and v is not None],
        )
    )
    DATASET_DIR: Path = PROJECT_ROOT / "data" / "encoded_data" / DATASET_KEY
    if not DATASET_DIR.exists() or not use_cached:
        train_dataset = load_dataset(
            dataset_params.name,
            split=dataset_params.train_split,
            use_auth_token=True,
        )
        test_dataset = load_dataset(dataset_params.name, split=dataset_params.test_split)
        dataset: DatasetDict = MyDatasetDict(train=train_dataset, test=test_dataset)
    else:
        dataset: Dataset = load_from_disk(dataset_path=str(DATASET_DIR))

    return dataset


def save_dataset_to_disk(dataset, output_path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    dataset.save_to_disk(output_path)


def preprocess_dataset(dataset, img_key, label_key, dataset_img_key, dataset_label_key):
    if dataset_label_key != label_key:
        dataset = dataset.rename_column(dataset_label_key, label_key)

    if dataset_img_key != img_key:
        dataset = dataset.rename_column(dataset_img_key, img_key)

    # in case some images are not RGB, convert them to RGB
    dataset = dataset.map(lambda x: {img_key: convert_to_rgb(x[img_key])}, desc="Converting to RGB")
    dataset.set_format(type="numpy", columns=[img_key, label_key])

    return dataset


def convert_dataset_to_rgb(dataset, img_key, label_key):
    # in case some images are not RGB, convert them to RGB
    dataset = dataset.map(lambda x: {img_key: convert_to_rgb(x[img_key])}, desc="Converting to RGB")

    dataset.set_format(type="numpy", columns=[img_key, label_key])

    return dataset


def add_ids_to_dataset(dataset):
    N = len(dataset["train"])
    M = len(dataset["test"])
    indices = {"train": list(range(N)), "test": list(range(N, N + M))}

    for mode in ["train", "test"]:
        dataset[mode] = dataset[mode].map(lambda row, ind: {"id": indices[mode][ind]}, with_indices=True)

    return dataset
