#modified based on: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
from PIL import Image #can solve the error of Glibc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from datautils import MyTrainDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import time
import os
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3" #"0,1"

MACHINENAME='Container'
USE_AMP=True #AUTOMATIC MIXED PRECISION
if MACHINENAME=='HPC':
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/'
    DATAPATH='/data/cmpe249-fa22/torchvisiondata/'
elif MACHINENAME=='Container':
    DATAPATH='/Dataset/Dataset/torchvisiondata'

#unused
class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

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
    
#added for multi-GPU processing 
import torch.multiprocessing as mp #a PyTorch wrapper around Python’s native multiprocessing
from torch.utils.data.distributed import DistributedSampler #DistributedSampler chunks the input data across all distributed processes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #initializes the distributed process group, The distributed process group contains all the processes that can communicate and synchronize with each other.
    #Different backend: https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = gpu_id #int(os.environ["LOCAL_RANK"])
        #self.model = model.to(gpu_id)
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if USE_AMP:
            self.scaler = torch.cuda.amp.GradScaler()

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.loss_fun = nn.CrossEntropyLoss() 
        #Constructing the DDP model
        
        self.model = DDP(model, device_ids=[gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        #loss = F.cross_entropy(output, targets)
        loss = self.loss_fun(output, targets)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        #print(f"loss: {loss:>7f}")
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        
        #Calling the set_epoch() method on the DistributedSampler at the beginning of each epoch is necessary to make shuffling work properly across multiple epochs. 
        # Otherwise, the same ordering will be used in each epoch.
        self.train_data.sampler.set_epoch(epoch)

        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            if USE_AMP:
                currentloss = self._run_batch_amp(source, targets)
            else:
                currentloss = self._run_batch(source, targets)
        print(f"loss: {currentloss:>7f}")
        return currentloss

    #“automatic mixed precision training” means training with torch.autocast and torch.cuda.amp.GradScaler together
    def _run_batch_amp(self, source, targets):
        self.optimizer.zero_grad()
        # Runs the forward pass with autocasting.
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(source)
            loss = self.loss_fun(output, targets)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        self.scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        self.scaler.step(self.optimizer)

        # Updates the scale for next iteration.
        self.scaler.update()

        loss = loss.item()
        #print(f"loss: {loss:>7f}")
        return loss

    def _save_checkpoint(self, epoch):
        #ckp = self.model.state_dict()
        ckp = self.model.module.state_dict()
        PATH = "./data"
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        torch.save(ckp, os.path.join(PATH, 'checkpoint.pt'))
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.gpu_id), y.to(self.gpu_id)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            #We only need to save model checkpoints from one process. 
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self.test(self.test_data, self.model, self.loss_fun)
                self._save_snapshot(epoch)
                #self._save_checkpoint(epoch)
        self.test(self.test_data, self.model, self.loss_fun)

# def load_train_objs():
#     train_set = MyTrainDataset(2048)  # load your dataset
#     model = torch.nn.Linear(20, 1)  # load your model
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     return train_set, model, optimizer

def load_train_objs():
    # Download training data from open datasets.
    #data_path="/data/cmpe249-fa22/torchvisiondata"
    train_set = datasets.FashionMNIST(
        root=DATAPATH,
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_set = datasets.FashionMNIST(
        root=DATAPATH,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # train_set = MyTrainDataset(2048)  # load your dataset
    #model = torch.nn.Linear(20, 1)  # load your model
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, test_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    # return DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     shuffle=True
    # )
    #Each process will receive an input batch of 32 samples; 
    # the effective batch size is 32 * nprocs, or 128 when using 4 GPUs.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) #chunks the input data across all distributed processes
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "/home/010796032/MyRepo/DeepDataMiningLearning/data/snapshot.pt"):
    #rank is auto-allocated by DDP when calling mp.spawn.
    #world_size is the number of processes across the training job. 
    # For GPU training, this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.
    ddp_setup(rank, world_size)

    train_dataset, test_dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)
    trainer = Trainer(model, train_data, test_data, optimizer, rank, save_every, snapshot_path)
    trainer.train(total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)