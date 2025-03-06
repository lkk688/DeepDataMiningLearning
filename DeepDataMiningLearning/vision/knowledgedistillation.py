import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, DefaultDataCollator, Trainer, TrainingArguments
import evaluate #pip install evaluate
from mydatasets import load_mydataset
from mymodels import load_mymodel

# Test function
def test(model, test_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def get_device(gpuid='0', useamp=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        useamp = False
    else:
        device = torch.device("cpu")
        useamp = False
    return device, useamp

# PyTorch training loop
def train_pytorch(model, train_loader, test_loader, args, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {running_loss / len(train_loader):.4f}")
        
def train_torch(teacher_model, student_model, train_loader, optimizer, epoch, temperature, alpha, device):
    """
    Unified training function for Knowledge Distillation.
    Works with both torchvision and Hugging Face models and datasets.
    
    Args:
        teacher_model: Teacher model (torchvision or Hugging Face).
        student_model: Student model (torchvision or Hugging Face).
        train_loader: DataLoader for the training dataset.
        optimizer: Optimizer for the student model.
        epoch: Current epoch number.
        temperature: Temperature for distillation.
        alpha: Weight for distillation loss.
        device: Device to use (e.g., "cuda" or "cpu").
    """
    teacher_model.eval()  # Teacher model is in eval mode
    student_model.train()  # Student model is in training mode

    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # Handle dataset differences
        if isinstance(batch, dict):  # Hugging Face dataset
            data = batch["pixel_values"].to(device)
            target = batch["label"].to(device)
        else:  # Torchvision dataset
            data, target = batch
            data, target = data.to(device), target.to(device)

        # Forward pass with teacher model
        with torch.no_grad():
            if hasattr(teacher_model, "logits"):  # Hugging Face model
                teacher_outputs = teacher_model(data).logits
            else:  # Torchvision model
                teacher_outputs = teacher_model(data)

        # Forward pass with student model
        if hasattr(student_model, "logits"):  # Hugging Face model
            student_outputs = student_model(data).logits
        else:  # Torchvision model
            student_outputs = student_model(data)

        # Compute losses
        loss_ce = nn.CrossEntropyLoss()(student_outputs, target)  # Standard cross-entropy loss
        loss_kd = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1)
        )  # Distillation loss

        # Combined loss
        loss = alpha * loss_ce + (1 - alpha) * loss_kd

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}], Average Loss: {running_loss / len(train_loader):.4f}")

# Hugging Face Trainer setup
def train_huggingface(model, train_dataset, test_dataset, args):
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
        return {"accuracy": acc["accuracy"]}
    training_args = TrainingArguments(
        output_dir="./output/hf_results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir="./output/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.lr,
        fp16=False,  # Enable mixed precision training, requires GPU
    )
    
    data_collator = DefaultDataCollator()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.evaluate()


def main():
    args = parse_args()
    # Load dataset
    train_dataset, test_dataset, num_classes, id2label, label2id = load_mydataset(args.dataset, args.data_dir, source=args.data_source, trainer=args.trainer, model_name=args.student_model)
    
    # Device configuration
    device, useamp = get_device()
    # Load models
    teacher_model = load_mymodel(args.teacher_model, num_classes, source=args.model_source).to(device)
    student_model = load_mymodel(args.student_model, num_classes, source=args.model_source).to(device)

    if args.trainer == "pytorch":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        # Optimizer
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
        # Training loop
        for epoch in range(args.epochs):
            train_torch(teacher_model, student_model, train_loader, optimizer, epoch, args.temperature, args.alpha, device)
            test(student_model, test_loader)
    elif args.trainer == "huggingface":
        train_huggingface(student_model, train_dataset, test_dataset, args)
    else:
        raise ValueError(f"Unsupported trainer: {args.trainer}")


def maintest():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    temperature = 4  # Temperature for softening logits
    alpha = 0.5  # Weight for distillation loss vs. student loss
    epochs = 10

    # Device configuration
    device, useamp = get_device()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets (CIFAR-10 as an example)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define teacher and student models
    teacher_model = models.resnet18(pretrained=True).to(device)
    student_model = models.mobilenet_v2(pretrained=False).to(device)  # Smaller model

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()  # Standard cross-entropy loss
    criterion_kd = nn.KLDivLoss(reduction='batchmean')  # Knowledge distillation loss

    # Optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    # Training loop
    def train_kd(teacher_model, student_model, train_loader, optimizer, epoch):
        teacher_model.eval()  # Teacher model is in eval mode
        student_model.train()  # Student model is in training mode

        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass with teacher model
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            # Forward pass with student model
            student_logits = student_model(data)

            # Compute losses
            loss_ce = criterion_ce(student_logits, target)  # Standard cross-entropy loss
            loss_kd = criterion_kd(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            )  # Distillation loss

            # Combined loss
            loss = alpha * loss_ce + (1 - alpha) * loss_kd

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}")


    # Training and testing
    for epoch in range(epochs):
        train_kd(teacher_model, student_model, train_loader, optimizer, epoch)
        test(student_model, test_loader, device)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation with PyTorch and Hugging Face")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="pcuenq/oxford-pets", choices=["cifar10", "cifar100", "imagenet"],
                        help="Dataset to use (cifar10, cifar100, imagenet)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store dataset")
    parser.add_argument("--data_source", type=str, default="huggingface", choices=["torchvision", "huggingface"],
                        help="Data source (torchvision or huggingface)")
    # Model arguments
    parser.add_argument("--teacher_model", type=str, default="asusevski/vit-base-patch16-224-oxford-pets",
                        help="Teacher model to use: resnet50")
    parser.add_argument("--student_model", type=str, default="WinKawaks/vit-tiny-patch16-224",
                        help="Student model to use: mobilenet_v2")
    parser.add_argument("--model_source", type=str, default="huggingface", choices=["torchvision", "huggingface"],
                        help="Model source (pytorch or huggingface)")
    # Training arguments
    parser.add_argument("--trainer", type=str, default="huggingface", choices=["pytorch", "huggingface"],
                        help="Training framework to use (pytorch or huggingface)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss")
    
    return parser.parse_args()

if __name__ == "__main__":
    #maintest()
    main()