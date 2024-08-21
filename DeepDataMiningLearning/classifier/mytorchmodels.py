# modified based on https://github.com/lkk688/MultiModalClassifier/blob/main/TorchClassifier/myTorchModels/TorchCNNmodels.py

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import time
import torchvision
from torchvision import datasets, models, transforms

# new approach: https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/
from torchvision.models import get_model, get_model_weights, get_weight, list_models

# print("Torch buildin models:", list_models())
model_names = list_models(module=torchvision.models)
# print("Torchvision buildin models:", model_names)
from timm.utils import AverageMeter
import timm  # pip install timm
import timm.optim
import timm.scheduler
# model_names = timm.list_models(pretrained=True)
# model_names = timm.list_models('*resnet*')

from mydataset import get_labelfn, timmget_datasetfromfolder, timmcreate_dataloader

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    # !pip install -q torchinfo
    # from torchinfo import summary


def create_torchclassifiermodel(
    model_name,
    numclasses=None,
    model_type="torchvision",
    torchhublink=None,
    freezeparameters=True,
    pretrained=True,
    dropoutp=0.2,
):
    pretrained_model = None
    preprocess = None
    imagenet_classes = None
    # different model_type: 'torchvision', 'torchhub'
    if model_type == "torchvision":
        model_names = list_models(module=torchvision.models)
        if model_name in model_names:
            # Step 1: Initialize model with the best available weights
            weights_enum = get_model_weights(model_name)
            weights = weights_enum.IMAGENET1K_V1
            # print([weight for weight in weights_enum])
            # weights = get_weight("ResNet50_Weights.IMAGENET1K_V2")#ResNet50_Weights.DEFAULT
            if pretrained == True:
                pretrained_model = get_model(
                    model_name, weights=weights
                )  # weights="DEFAULT"
                # pretrained_model=get_model(model_name, weights="DEFAULT")
            else:
                pretrained_model = get_model(model_name, weights=None)
            # print(pretrained_model)
            # Freeze the base parameters
            if freezeparameters == True:
                print("Freeze parameters")
                for parameter in pretrained_model.parameters():
                    parameter.requires_grad = False
            # Step 2: Initialize the inference transforms
            preprocess = weights.transforms()  # preprocess.crop_size
            imagenet_classes = weights.meta["categories"]
            # Step 3: Apply inference preprocessing transforms
            # batch = preprocess(img).unsqueeze(0)
            if numclasses is not None and len(imagenet_classes) != numclasses:
                pretrained_model = modify_classifier(
                    pretrained_model=pretrained_model,
                    numclasses=numclasses,
                    dropoutp=dropoutp,
                )
            else:
                numclasses = len(imagenet_classes)
            return pretrained_model
        else:
            print("Model name not exist.")
    elif model_type == "torchhub" and torchhublink is not None:
        #'deit_base_patch16_224'
        # pretrained_model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        pretrained_model = torch.hub.load(
            torchhublink, model_name, pretrained=pretrained
        )
    elif model_type == "timm":
        # if model_name in model_names:
        pretrained_model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=numclasses
        )
        data_cfg = timm.data.resolve_data_config(pretrained_model.pretrained_cfg)
        preprocess = timm.data.create_transform(**data_cfg)

    totalparameters= sum([m.numel() for m in pretrained_model.parameters()])
    print(
        f"Model {model_name} created, param count: {totalparameters/10e6}M"
    )
    num_classes = getattr(pretrained_model, "num_classes", None)
    return pretrained_model, preprocess, num_classes, imagenet_classes


def modify_classifier(pretrained_model, numclasses, dropoutp=0.3, classifiername=None):
    # display model architecture
    lastmoduleinlist = list(pretrained_model.named_children())[-1]
    # print("lastmoduleinlist len:",len(lastmoduleinlist))
    lastmodulename = lastmoduleinlist[0]
    print("lastmodulename:", lastmodulename)
    lastlayer = lastmoduleinlist[-1]
    if isinstance(lastlayer, nn.Linear):
        print("Linear layer")
        newclassifier = nn.Linear(
            in_features=lastlayer.in_features, out_features=numclasses
        )
    elif isinstance(lastlayer, nn.Sequential):
        print("Sequential layer")
        lastlayerlist = list(lastlayer)  # [-1] #last layer
        # print("lastlayerlist type:",type(lastlayerlist))
        if isinstance(lastlayerlist, list):
            # print("your object is a list !")
            lastlayer = lastlayerlist[-1]
            newclassifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropoutp, inplace=True),
                torch.nn.Linear(
                    in_features=lastlayer.in_features,
                    out_features=numclasses,  # same number of output units as our number of classes
                    bias=True,
                ),
            )
        else:
            print("Error: Sequential layer is not list:", lastlayer)
            # newclassifier = nn.Linear(in_features=lastlayer.in_features, out_features=classnum)
    if lastmodulename == "heads":
        pretrained_model.heads = newclassifier  # .to(device)
    elif lastmodulename == "classifier":
        pretrained_model.classifier = newclassifier  # .to(device)
    elif lastmodulename == "fc":
        pretrained_model.fc = newclassifier  # .to(device)
    elif classifiername is not None:
        lastlayer = newclassifier  # not tested!!
    else:
        print("Please check the last module name of the model.")

    return pretrained_model


def get_optimizer(model, opt="lamb", lr=0.01, weight_decay=0.01, momentum=0):
    # optimizer = timm.optim.AdamP(model.parameters(), lr=0.01)
    # opt: name of optimizer to create: sgd, adam, adamw, adamp, adabelief, lamp
    # lr: initial learning rate
    # weight_decay: weight decay to apply in optimizer
    # momentum:  momentum for momentum based optimizers
    optimizer = timm.optim.create_optimizer_v2(
        model, opt=opt, lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    return optimizer


def get_scheduler(optimizer, num_epochs, num_repeat=2, warmup_t=3):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                                 T_0=num_epoch_repeat*num_steps_per_epoch,
    #                                                                 T_mult=1,
    #                                                                 eta_min=1e-6,
    #                                                                 last_epoch=-1)
    num_epoch_repeat = num_epochs // num_repeat
    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer,
        t_initial=num_epoch_repeat,
        lr_min=1e-5,
        warmup_lr_init=0.01,
        warmup_t=warmup_t,
        cycle_limit=num_epoch_repeat + 1,
    )
    return scheduler


def train_step(
    model, dataloader, loss_fn, optimizer, device, epoch, scheduler, ema_model
):
    """Trains a PyTorch model for a single epoch.
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    num_steps_per_epoch = len(dataloader)
    num_updates = epoch * num_steps_per_epoch

    lrs = []

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Update EMA model parameters
        if ema_model is not None:
            ema_model.update(model)

        # new added for timm
        if scheduler is not None:
            num_updates += 1
            scheduler.step_update(num_updates=num_updates)
            lrs.append(optimizer.param_groups[0]["lr"])

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, lrs, ema_model


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    epochs,
    device,
    scheduler,
    use_ema=False,
):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary

    # Loop through training and testing steps for a number of epochs

    # Make sure model on target device
    model.to(device)

    if use_ema:
        ema_model = timm.utils.ModelEmaV2(model, decay=0.9)
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "ematest_loss": [],
            "ematest_acc": [],
            "all_lrs": [],
        }
    else:
        ema_model = None
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "all_lrs": [],
        }

    all_lrs = []

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, lrs, ema_model = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            ema_model=ema_model,
        )

        if scheduler is not None:
            all_lrs.extend(lrs)
            scheduler.step(epoch + 1)

        test_loss, test_acc, _, _, _ = inference(
            model=model, dataloader=test_dataloader, device=device, loss_fn=loss_fn, use_probs=False, top_k=None, to_label=None
        )
        if ema_model is not None:
            ematest_loss, ematest_acc, _, _, _ = inference(
                model=ema_model,
                dataloader=test_dataloader,
                device=device,
                loss_fn=loss_fn,
                use_probs=False, top_k=None, to_label=None
            )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        if ema_model is not None:
            print( f"test_loss: {ematest_loss:.4f}, test_acc: {ematest_acc:.4f}"
            )
            

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if ema_model is not None:
            results["ematest_loss"].append(ematest_loss)
            results["ematest_acc"].append(ematest_acc)
        results["all_lrs"].append(all_lrs)

    # Return the filled results at the end of the epochs
    return results, ema_model


def inference(
    model, dataloader, device, loss_fn=None, use_probs=True, top_k=5, to_label=None
):
    # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    num_classes = getattr(model, "num_classes", 1000)

    if top_k is not None:
        top_k = min(top_k, num_classes)
    all_indices = []
    all_labels = []
    all_outputs = []
    to_label = get_labelfn(model)
    batch_time = AverageMeter()
    end = time.time()
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            if y is not None:
                # Send data to target device
                X, y = X.to(device), y.to(device)
            else:
                X = X.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            if loss_fn is not None:
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            if y is not None:
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            if use_probs:
                output = test_pred_logits.softmax(-1)
            else:
                output = test_pred_logits

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    return test_loss, test_acc, all_indices, all_labels, all_outputs

def test_train():
    model_name = "resnet50"  #'mobilenetv3_large_100' 'resnet50' 'resnest26d'
    model, preprocess, num_classes, imagenet_classes = create_torchclassifiermodel(
        model_name,
        numclasses=None,
        model_type="timm",
        freezeparameters=False,
        pretrained=True,
    )
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    model = model.to(device)
    model.eval()
    
    data_path='data/imagenette/imagenette2-320'
    train_data, test_data, num_classes, class_names = \
        timmget_datasetfromfolder(data_path=data_path, mapidfile="data/map_clsloc.txt", use_autoaugment=False)

    batch_size = 16
    num_epochs = 20
    train_dataloader, test_dataloader = timmcreate_dataloader(model, train_data, test_data, batch_size=batch_size)
    
    optimizer = get_optimizer(model=model, opt='adamw')
    loss_fn = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer=optimizer, num_epochs=num_epochs, num_repeat=2)
    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # Setup training and save the results
    results, ema_model = train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=num_epochs,
                        device=device,
                        scheduler=scheduler,
                        use_ema = True)
    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    #https://pytorch.org/tutorials/beginner/saving_loading_models.html
    os.makedirs('data', exist_ok=True)
    filename=os.path.join('data', 'modelcheckpoint.pth.tar')
    torch.save(model.state_dict(), filename)
    filename=os.path.join('data', 'emamodelcheckpoint.pth.tar')
    torch.save(ema_model.state_dict(), filename)
    
    # Saving the data
    np.save("data/results.npy", results)
    # # Loading the data
    # data = np.load("d.npy",allow_pickle=True)
    # # Display dict items
    # print("Dict items:\n",data.item().get('B'))
    return results

def test_createmodel():
    #model = torch.jit.script(model)

    model_name = "efficientnet_b1"
    model = create_torchclassifiermodel(
        model_name,
        numclasses=None,
        model_type="torchvision",
        freezeparameters=False,
        pretrained=True,
    )
    summary(
        model=model,
        input_size=(
            32,
            3,
            224,
            224,
        ),  # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

if __name__ == "__main__":
    test_train()

    