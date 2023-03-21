#https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from datautils import MyTrainDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import time
import os

from PIL import Image

import torchvision

MACHINENAME='HPC'
if MACHINENAME=='HPC':
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/'
    DATAPATH='/data/cmpe249-fa22/torchvisiondata/'
elif MACHINENAME=='AlienwareX51':
    DATAPATH='/DATA8T/Datasets/torchvisiondata'

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.loss_fun = nn.CrossEntropyLoss() 

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        #loss = F.cross_entropy(output, targets)
        loss = self.loss_fun(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def load_train_objs():
    # Download training data from open datasets.
    data_path="/data/cmpe249-fa22/torchvisiondata/MNIST"
    train_set = datasets.MNIST(
        root=data_path,
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_set = datasets.MNIST(
        root=data_path,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    main(device, args.total_epochs, args.save_every, args.batch_size)