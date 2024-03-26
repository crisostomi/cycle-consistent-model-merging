import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class HuggingFaceTorchVisionDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset (datasets.Dataset): The Hugging Face dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.hf_dataset[idx]["image"]
        label = self.hf_dataset[idx]["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TinyImageNet(HuggingFaceTorchVisionDataset):
    def __init__(self, ref, split):

        hf_dataset = load_dataset(
            ref,
            split=split,
            use_auth_token=True,
        )

        # hf_dataset = convert_dataset_to_rgb(hf_dataset, img_key="image", label_key="label")

        super().__init__(hf_dataset)

        # self.num_classes = num_classes
