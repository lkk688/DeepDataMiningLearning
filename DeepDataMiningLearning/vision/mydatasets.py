import os
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision import datasets as torchdatasets
from torch.utils.data import DataLoader
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetInfo, DatasetDict, Features, Image as ImageFeature, ClassLabel
from datasets import load_dataset as hf_load_dataset
from transformers import default_data_collator, AutoImageProcessor, AutoConfig
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


def test_torch2hfdataset():
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



def bytes_to_pil(image_bytes):
    """
    Converts image data in bytes format to a PIL Image object.

    Args:
        image_bytes (bytes): The image data in bytes format.

    Returns:
        PIL.Image.Image: The PIL Image object, or None if an error occurs.
    """
    try:
        # Use BytesIO to create an in-memory file-like object from the bytes.
        image_stream = BytesIO(image_bytes)

        # Open the image using PIL's Image.open()
        pil_image = Image.open(image_stream)

        return pil_image
    except Exception as e:
        print(f"Error converting bytes to PIL Image: {e}")
        return None
    
def tensor_to_pil(tensor, denormalize=True):
    """
    Converts a PyTorch tensor to a PIL Image.

    Args:
        tensor (torch.Tensor): The input tensor. It is assumed to be in CHW format (channels, height, width).
        denormalize (bool, optional): Whether to denormalize the tensor if it was normalized. Defaults to True.

    Returns:
        PIL.Image.Image: The converted PIL Image.
    """

    # Check if the input is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input should be a PyTorch tensor.")

    # Check if the tensor is in CHW format
    if tensor.ndim != 3:
        raise ValueError("Input tensor should be 3-dimensional (CHW).")

    # Check if channels are valid, either 1 (grayscale) or 3 (color).
    if tensor.shape[0] not in (1, 3):
        raise ValueError("Input tensor should have 1 or 3 channels.")

    # Clone the tensor to avoid modifying the original tensor
    tensor = tensor.clone().detach().cpu()

    # Denormalize the tensor if necessary
    if denormalize:
        # This is a common normalization scheme, adjust if yours is different
        mean = torch.tensor([0.5, 0.5, 0.5]) if tensor.shape[0] == 3 else torch.tensor([0.5])
        std = torch.tensor([0.5, 0.5, 0.5]) if tensor.shape[0] == 3 else torch.tensor([0.5])
        tensor = tensor * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

    # Clamp values to the valid range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL Image
    #PIL expects pixel values to be in the range of [0, 255] if you are creating the image directly from numpy array. 
    # However, if you create image using transforms.ToPILImage, it expects the pixel values are in the range of [0, 1] or [0,255].
    if tensor.shape[0] == 1:  # Grayscale
        image = transforms.ToPILImage(mode="L")(tensor)
    else:  # Color
        image = transforms.ToPILImage()(tensor)

    return image


def load_mydataset(dataset_name, data_dir, source="torchvision", trainer="huggingface",image_size=None,model_name=None):
    """
    Load dataset from either torchvision or Hugging Face datasets library.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "cifar10", "imagenet", "cifar100").
        data_dir (str): Directory to store/load the dataset.
        source (str): Source of the dataset ("torchvision" or "huggingface").
        model_name (str): Name of the model (required if source is "huggingface" or to infer image size for torchvision).
    
    Returns:
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        num_classes: Number of classes in the dataset.
    """
    
    if source == "torchvision":
        image_size = image_size if image_size else (224, 224)
        transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize for compatibility with ViT
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if dataset_name == "cifar10":
            train_dataset = torchdatasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = torchdatasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
            num_classes = 10
        elif dataset_name == "cifar100":
            train_dataset = torchdatasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = torchdatasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
            num_classes = 100
        elif dataset_name == "imagenet":
            train_dataset = torchdatasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
            test_dataset = torchdatasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
            num_classes = 1000
        else:
            raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")
        class_names = train_dataset.classes
        num_classes = len(class_names)
        # id2label: Mapping from class index to class name
        id2label = {i: class_names[i] for i in range(num_classes)}
        # label2id: Mapping from class name to class index
        label2id = {class_names[i]: i for i in range(num_classes)}
        # Get one sample, image is CHW tensor, label is int label
        image, label = train_dataset[0]
        # Save the image
        image_pil = tensor_to_pil(image)
        image_pil.save('./output/torchimage.png')
        
    elif source == "huggingface":
        if dataset_name == "cifar10":
            dataset = hf_load_dataset("cifar10")
            num_classes = 10
        elif dataset_name == "cifar100":
            dataset = hf_load_dataset("cifar100")
            num_classes = 100
        elif dataset_name == "imagenet-1k":
            dataset = hf_load_dataset("imagenet-1k")
            num_classes = 1000
        else:
            dataset = hf_load_dataset(dataset_name)
            #num_classes = dataset['train'].info.features["label"].num_classes

        id2label = {int_label: str_label for int_label, str_label in enumerate(sorted(list(set(dataset['train']['label']))))}
        label2id = {str_label: int_label for int_label, str_label in enumerate(sorted(list(set(dataset['train']['label']))))}
        num_classes = len(id2label)
        image = dataset["train"][0]['image']['bytes'] #bytes
        bytes_to_pil(image).save("./output/hfimage.png")
        
        # Convert Hugging Face dataset to PyTorch dataset
        def apply_torchtransform(examples):
            examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["img"]]
            return examples
        
        # Load Hugging Face processor
        if model_name and image_size:
            processor = AutoImageProcessor.from_pretrained(model_name, size=image_size)
        elif model_name:
            processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            processor = None
        
        # Preprocess function for Hugging Face dataset
        def hfpreprocess_fn(examples):
            # Convert images to RGB and preprocess using the processor
            if isinstance(examples['image'], Image.Image):
                images = [image.convert("RGB") for image in examples["image"]]
            else:
                images = [bytes_to_pil(image['bytes']).convert("RGB") for image in examples["image"]]
            
            #inputs = processor(images, return_tensors="pt")
            #examples["pixel_values"] = inputs["pixel_values"]
            examples["pixel_values"] = [
                processor(image, return_tensors="pt") for image in images
            ]
            #examples['label'] = label2id[examples['label']]
            examples['label'] = [int(label2id[y]) for y in examples['label']]
            return examples
        
        import io
        def transforms(example):
            image = Image.open(io.BytesIO(example['image']['bytes']))
            # Convert to RGB - there are some example in the Oxford Pets dataset that are RGBA, and even at least one gif
            example['image'] = image.convert('RGB')

            # Feed image into ViT image processor
            inputs = processor(example['image'], return_tensors='pt')

            # Add to example
            example['pixel_values'] = inputs['pixel_values'].squeeze() #[1, 3, 224, 224]=>[3,244,244]
            example['label'] = label2id[example['label']] # int
            return example
        
        if source == "huggingface" and processor:
            #dataset = dataset.with_transform(hfpreprocess_fn)
            dataset = dataset.map(transforms)
            #dataset = dataset.map(hfpreprocess_fn, batched=True, remove_columns=["img"])
        else:
            dataset = dataset.map(apply_torchtransform, batched=True, remove_columns=["img"])
        
        if trainer == "huggingface":
            dataset.set_format("pt", columns=["pixel_values"], output_all_columns=True)
        else:
            dataset.with_format("torch")
        #split_datasets = dataset["train"].train_test_split(test_size=0.15, seed=20)
        train_test_dataset = dataset['train'].train_test_split(test_size=0.2)
        train_val_dataset = train_test_dataset['train'].train_test_split(test_size=(0.1/0.8))
        dataset_dict = DatasetDict({
            'train': train_val_dataset['train'],
            'valid': train_val_dataset['test'],
            'test': train_test_dataset['test']
        })
        train_dataset = train_val_dataset['train']
        test_dataset = train_test_dataset['test']
        # if trainer == "huggingface":
            
        #     split_datasets = dataset["train"].train_test_split(test_size=0.15, seed=20)
        #     train_dataset = split_datasets["train"]
        #     test_dataset = split_datasets["test"]
        # else:
        #     split_datasets = dataset["train"].train_test_split(test_size=0.15, seed=20)
        #     train_dataset = split_datasets["train"].with_format("torch")
        #     test_dataset = split_datasets["test"].with_format("torch")
    
    else:
        raise ValueError(f"Unsupported dataset source: {source}")
    
    
    return train_dataset, test_dataset, num_classes, id2label, label2id

# Testing code for the dataset
def test_dataset_loading():
    # Test torchvision dataset
    print("Testing torchvision CIFAR-10 dataset...")
    train_dataset, test_dataset, num_classes = load_mydataset("cifar10", "./data", source="torchvision")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Sample image shape: {train_dataset[0][0].shape}")
    print(f"Sample label: {train_dataset[0][1]}")
    print()
    
    # Test Hugging Face dataset
    print("Testing Hugging Face CIFAR-10 dataset...")
    train_dataset, test_dataset, num_classes = load_mydataset("cifar10", "./data", source="huggingface")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Sample image shape: {train_dataset[0]['pixel_values'].shape}")
    print(f"Sample label: {train_dataset[0]['label']}")
    
if __name__ == "__main__":
    test_dataset_loading()
    test_torch2hfdataset()