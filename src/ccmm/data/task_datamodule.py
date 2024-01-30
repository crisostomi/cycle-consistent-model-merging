import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional, Union

import pytorch_lightning as pl
from datasets import concatenate_datasets
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from nn_core.nn_types import Split

from ccmm.data.my_dataset_dict import MyDatasetDict

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, tasks_info: Mapping[str, int]):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        """
        self.tasks_info: Mapping[str, int] = tasks_info

    def save(self, dst_path: Path) -> None:
        """
        Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        (dst_path / "tasks_info.tsv").write_text("\n".join(f"{key}\t{value}" for key, value in self.tasks_info.items()))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        lines = (src_path / "tasks_info.tsv").read_text(encoding="utf-8").splitlines()

        tasks_info = {}
        for line in lines:
            key, value = line.strip().split("\t")
            tasks_info[key] = value

        return MetaData(
            tasks_info=tasks_info,
        )


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        data_path: Path,
        only_use_sample_num: int = -1,
        keep_img_column: bool = False,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.pin_memory = False

        self.keep_img_column = keep_img_column

        # each mode will have multiple tasks
        self.datasets = {"train": {}, "val": {}, "test": {}}

        self.data: MyDatasetDict = MyDatasetDict.load_from_disk(dataset_dict_path=str(data_path))
        self.data.set_format("numpy", columns=["img", "y"])

        self.img_size = self.data["task_0_train"]["img"][0].shape[1]

        self.tasks = [key for key in self.data.keys() if key != "metadata"]
        self.num_tasks = self.data["metadata"]["num_tasks"]

        self.task_ind = None  # will be set in setup
        self.transform_func = None  # will be set in setup
        self.shuffle_train = True

        self.has_coarse_label = "coarse_label" in self.data[f"task_{0}_test"].features

        self.only_use_sample_num = only_use_sample_num

        if only_use_sample_num >= 0:
            self.data.select(range(only_use_sample_num))

        pylogger.info("Preprocessing done.")

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """

        return MetaData(tasks_info=self.data["metadata"])

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"][self.task_ind],
            shuffle=self.shuffle_train,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"][self.task_ind],
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"][self.task_ind],
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"

    def dataloader(self, mode, only_task_specific_test=False):
        if mode == "train":
            return self.train_dataloader()
        elif mode == "val":
            return self.val_dataloader()
        elif mode == "test":
            # at embedding time, we only want to embed the task-specific test samples
            # if only_task_specific_test:
            # test_dataloader = self.test_dataloader()
            # return test_dataloader[0] if isinstance(test_dataloader, list) else test_dataloader
            return self.test_dataloader()
        elif mode == "anchors":
            return self.anchor_dataloader()
        else:
            raise ValueError(f"Mode {mode} not supported")


class SameClassesDisjSamplesDatamodule(MyDataModule):
    def __init__(
        self,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        data_path: Path,
        only_use_sample_num: int = -1,
        train_on_anchors: bool = False,
    ):
        super().__init__(
            num_workers=num_workers,
            batch_size=batch_size,
            gpus=gpus,
            data_path=data_path,
            only_use_sample_num=only_use_sample_num,
        )

        # all tasks will have the same anchors
        for task_ind in range(self.num_tasks + 1):
            self.data[f"task_{task_ind}_anchors"] = self.data["anchors"]

        self.datasets = {"train": {}, "val": {}, "test": {}, "anchors": {}}

        self.train_on_anchors = train_on_anchors
        self.seen_tasks = set()

    def setup(self, stage: Optional[str] = None) -> None:
        # to avoid reprocessing the data
        if self.task_ind in self.seen_tasks:
            return

        self.shuffle_train = True

        map_params = {
            "function": lambda x: {"x": self.transform_func(x["img"])},
            "writer_batch_size": 100,
            "num_proc": 1,
        }

        modes = ["train", "val", "test", "anchors"]

        for mode in modes:
            self.data[f"task_{self.task_ind}_{mode}"] = self.data[f"task_{self.task_ind}_{mode}"].map(
                desc=f"Transforming task {self.task_ind} {mode} samples", **map_params
            )

            self.data[f"task_{self.task_ind}_{mode}"].set_format(type="torch", columns=["x", "y"])
            self.datasets[mode][self.task_ind] = self.data[f"task_{self.task_ind}_{mode}"]

        if self.train_on_anchors:
            self.datasets["train"][self.task_ind] = concatenate_datasets(
                self.datasets["train"], self.datasets["anchors"]
            )

        self.seen_tasks.add(self.task_ind)

    def test_dataloader(self) -> List[DataLoader]:
        test_dataloader_params = {
            "shuffle": False,
            "batch_size": self.batch_size.test,
            "num_workers": self.num_workers.test,
            "collate_fn": partial(collate_fn, split="test", metadata=self.metadata),
        }

        task_specific_dataloader = DataLoader(
            self.datasets["test"][self.task_ind],
            **test_dataloader_params,
        )

        dataloaders = [task_specific_dataloader]

        if self.task_ind >= 1:
            global_dataloader = DataLoader(
                self.datasets["test"][0],
                **test_dataloader_params,
            )
            dataloaders.append(global_dataloader)

        return dataloaders

    def anchor_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["anchors"][self.task_ind],
            shuffle=False,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="anchors", metadata=self.metadata),
        )
