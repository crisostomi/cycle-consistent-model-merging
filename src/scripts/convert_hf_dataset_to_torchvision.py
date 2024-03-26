import os

from datasets import load_dataset

from nn_core.common import PROJECT_ROOT

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

base_dir = f"{PROJECT_ROOT}/data" + dataset_name

datasets = {"train": train_hf_dataset, "test": test_hf_dataset}

for split in splits:
    classes = datasets[split].features["label"].names

    for class_name in classes:
        print("Creating class")
        os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

    for i, example in enumerate(datasets[split]):
        image = example["image"]
        label = example["label"]
        class_name = classes[label]

        image_path = os.path.join(base_dir, split, class_name, f"{i}.jpg")
        image.save(image_path)
