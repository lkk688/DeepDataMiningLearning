from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    ImageNetInfo,
    infer_imagenet_subset,
)
import numpy as np
import timm
from timm.data import ImageDataset
from timm.data import create_dataset
from timm.data.transforms_factory import create_transform
from pathlib import Path
import os
import pandas as pd

#Download dataset:
#~/Developer/DeepDataMiningLearning/data$ wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -P imagenette
#tar zxf imagenette/imagenette2-320.tgz -C imagenette

def timmget_datasetfromfolder(data_path, mapidfile="map_clsloc.txt", use_autoaugment=False, trainfoldername='train', valfoldername='val'):
    data_path = Path(data_path)
    train_path = data_path / trainfoldername
    val_path = data_path / valfoldername
    img_folder_paths = [folder for folder in train_path.iterdir() if folder.is_dir()]
    # Display the names of the folders using a Pandas DataFrame
    pd.DataFrame({"Image Folder": [folder.name for folder in img_folder_paths]})
    print(pd)

    # option1
    # num_classes = len(list(train_path.iterdir()))
    # option2
    classes = [
        d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))
    ]
    classes.sort()
    num_classes = len(classes)
    if mapidfile is not None:
        df = pd.read_csv(
            mapidfile,
            delimiter=" ",
            header=None,
            names=["WordNetID", "ID", "description"],
        )
        # use the WordNetID as the key
        id_to_desc = dict(zip(df["WordNetID"], df["description"]))
        class_names = [id_to_desc[class_name] for class_name in classes]
    else:
        class_names = classes
    
    if use_autoaugment:
        train_data = create_dataset(
            name="",
            root=train_path,
            transform=create_transform(
                224, is_training=True, auto_augment="rand-m9-mstd0.5"
            ),
        )
    else:
        train_data = create_dataset(
            name="",
            root=train_path,
            transform=create_transform(224, is_training=True),
        )
    test_data = create_dataset(
        name="", root=val_path, transform=create_transform(224)
    )
    return train_data, test_data, num_classes, class_names


def get_labelfn(model, label_type="name"):
    to_label = None
    label_type = "name"
    imagenet_subset = infer_imagenet_subset(model)  #'imagenet-1k'
    if imagenet_subset is not None:
        dataset_info = ImageNetInfo(imagenet_subset)
        if label_type == "name":
            to_label = lambda x: dataset_info.index_to_label_name(x)
        elif label_type == "detail":
            to_label = lambda x: dataset_info.index_to_description(x, detailed=True)
        else:
            to_label = lambda x: dataset_info.index_to_description(x)
        to_label = np.vectorize(to_label)
    return to_label, imagenet_subset


if __name__ == "__main__":
    # from mytorchmodels import create_torchclassifiermodel
    import timm

    model_name = "resnet50"  #'mobilenetv3_large_100' 'resnet50' 'resnest26d'
    # model, preprocess, numclasses, imagenet_classes = create_torchclassifiermodel(model_name, numclasses=None, model_type='timm', freezeparameters=False, pretrained=True)
    model = timm.create_model(model_name, pretrained=True, scriptable=True)
    model.eval()
    print(
        f"Model {model_name} created, param count: {sum([m.numel() for m in model.parameters()])}"
    )
    print(model.num_classes)

# root_dir = args.data or args.data_dir
#     dataset = create_dataset(
#         root=root_dir,
#         name=args.dataset,
#         split=args.split,
#         class_map=args.class_map,
#     )
