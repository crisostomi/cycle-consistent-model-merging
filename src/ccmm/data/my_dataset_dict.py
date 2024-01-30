import json
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Union

from datasets import DatasetDict, load_dataset

CUSTOM_METADATA_KEY = "metadata"


class MyDatasetDict(DatasetDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keys(self):
        return [k for k in super().keys() if k != CUSTOM_METADATA_KEY]

    def values(self):
        return [v for k, v in super().items() if k != CUSTOM_METADATA_KEY]

    def items(self):
        return [(k, v) for k, v in super().items() if k != CUSTOM_METADATA_KEY]

    def __iter__(self):
        for k in super().keys():
            if k != CUSTOM_METADATA_KEY:
                yield k

    def shard(self, num_shards, index) -> "DatasetDict":
        # DatasetDict is missing the shard method somehow ?
        return DatasetDict({k: dataset.shard(num_shards=num_shards, index=index) for k, dataset in self.items()})

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        fs="deprecated",
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[Dict[str, int]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        Path(dataset_dict_path).mkdir(exist_ok=True, parents=True)
        self.save_metadata(Path(dataset_dict_path) / f"{CUSTOM_METADATA_KEY}.json")
        super().save_to_disk(
            dataset_dict_path,
            fs=fs,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )

    def save_metadata(self, metadata_path: PathLike):
        with open(metadata_path, "w") as f:
            json.dump(self[CUSTOM_METADATA_KEY], f)

    @staticmethod
    def load_from_disk(
        dataset_dict_path: PathLike,
        fs="deprecated",
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "MyDatasetDict":

        dataset = MyDatasetDict(
            DatasetDict.load_from_disk(
                dataset_dict_path,
                fs=fs,
                keep_in_memory=keep_in_memory,
                storage_options=storage_options,
            )
        )
        dataset[CUSTOM_METADATA_KEY] = json.load(open(Path(dataset_dict_path) / f"{CUSTOM_METADATA_KEY}.json"))

        return dataset

    def select(self, indices):
        for k, v in self.items():
            self[k] = v.select(indices)


if __name__ == "__main__":

    train_dataset = load_dataset(
        "cifar100",
        split="train",
        use_auth_token=True,
    )
    test_dataset = load_dataset("cifar100", split="test")

    my_dataset = MyDatasetDict(train=train_dataset, test=test_dataset)

    assert my_dataset.keys() == ["train", "test"]

    tmp_path = Path("tmp/")
    tmp_path.mkdir(exist_ok=True)

    my_dataset["metadata"] = {"num_classes": 100, "model": None}

    my_dataset.save_to_disk(tmp_path)

    assert my_dataset.keys() == ["train", "test"]

    my_dataset = MyDatasetDict.load_from_disk(tmp_path)

    assert my_dataset.keys() == ["train", "test"]

    my_dataset = my_dataset.map(lambda x: {"new_column": x["fine_label"] * 2})

    assert my_dataset["train"]["new_column"][0] == my_dataset["train"]["fine_label"][0] * 2
    assert my_dataset["test"]["new_column"][0] == my_dataset["test"]["fine_label"][0] * 2

    my_dataset = my_dataset.filter(lambda x: x["fine_label"] < 10)

    my_dataset.set_format("numpy", columns=["fine_label"])
    assert (my_dataset["train"]["fine_label"] < 10).all()
    assert (my_dataset["test"]["fine_label"] < 10).all()
