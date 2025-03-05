import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

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


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    temperature = 4  # Temperature for softening logits
    alpha = 0.5  # Weight for distillation loss vs. student loss
    epochs = 10

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        test(student_model, test_loader)

if __name__ == "__main__":
    main()