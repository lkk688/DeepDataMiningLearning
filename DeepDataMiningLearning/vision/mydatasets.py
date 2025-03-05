import os
from PIL import Image
from torchvision import transforms
from torchvision import datasets as torchdatasets
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetInfo, Features, Image as ImageFeature, ClassLabel
from transformers import default_data_collator
import torch

def image_generator(torchvision_dataset=None, local_folder=None, transform=None):
    print("image_generator called!")
    if torchvision_dataset:
        for img, target in torchvision_dataset:
            try:
                if transform:
                    img = transform(img)
                img = transforms.ToTensor()(img) #Ensure to convert to tensor
                yield {"image": img, "label": target}
            except Exception as e:
                print(f"Error processing torchvision image: {e}")
                print(f"Image: {img}, Target: {target}")
    elif local_folder:
        for class_name in os.listdir(local_folder):
            class_path = os.path.join(local_folder, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        if transform:
                            img = transform(img)
                        yield {"image": img, "label": class_name}
                    except Exception as e:
                        print(f"Error loading local image {img_path}: {e}")


def main():
    # Example torchvision dataset (MNIST)
    torchvision_dataset = torchdatasets.MNIST(root="./data", train=True, download=True)

    # Example local image folder (replace with your folder path)
    local_folder = "./local_images"
    os.makedirs(local_folder, exist_ok=True)

    # Example transform
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    transform = transforms.ToTensor()

    # Define features
    # features = Features({
    #     "image": ImageFeature(),
    #     "label": ClassLabel(names=sorted(list(set(str(target) for _, target in torchvision_dataset) | set(os.listdir(local_folder) if os.path.isdir(local_folder) else []))))
    # })
    features = Features({
        "image": ImageFeature(),
        "label": ClassLabel(names=sorted(list(set(str(target) for _, target in torchvision_dataset) )))
    })

    # Define dataset info
    dataset_info = DatasetInfo(
        description="Combined torchvision and local image dataset",
        features=features,
        supervised_keys=("image", "label")
    )

    # Create the Hugging Face Dataset
    try:
        hfdataset = HuggingFaceDataset.from_generator(
            image_generator,
            gen_kwargs={
                "torchvision_dataset": torchvision_dataset,
                "local_folder": None,
                "transform": None
            },
            features=features,
            info=dataset_info
        )
    except Exception as e:
        print(f"Error during dataset generation: {e}")

    # Test the dataset
    #image_generator yields dictionaries like {"image": tensor, "label": label}.
    #The DataLoader needs to combine these individual samples into batches. By default, when it encounters dictionaries with tensors, it groups the tensors with the same key into lists. So, batch["image"] becomes a list of tensors, and batch["label"] becomes a list of labels.
    print(hfdataset[0].keys())
    print(type(hfdataset[10]['image']))#PIL
    print(hfdataset[0])
    print(hfdataset.features) #{'image': Image(mode=None, decode=True, id=None), 'label': ClassLabel(names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], id=None)}

    # Create a DataLoader for testing with default_data_collator
    #To get batch["image"] as a tensor, you need to use a custom collation function in your DataLoader. Hugging Face Datasets provides a handy default_data_collator that does exactly what you need.
    dataloader = torch.utils.data.DataLoader(
        hfdataset,
        batch_size=4,
        shuffle=True,
        collate_fn=default_data_collator #This is the change.
    )

    # Iterate through the DataLoader
    for batch in dataloader:
        print("Images shape:", batch["image"].shape)
        print("Labels:", batch["labels"])
        break

if __name__ == "__main__":
    main()