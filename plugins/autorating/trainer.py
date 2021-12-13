import logging
import os
import time
from typing import Tuple
from pathlib import Path
import torch
import torch.optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .common import Transform
from .dataset import TLDataset
from .model import NIMA, create_model, EDMLoss


logger = logging.getLogger(__file__)


def get_dataloaders(batch_size: int, num_workers: int, working_dir : str = '') -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = Transform()

    train_ds = TLDataset(os.path.join(working_dir, 'train'), transform.train_transform)
    val_ds = TLDataset(os.path.join(working_dir, 'val'), transform.val_transform)
    test_ds = TLDataset(os.path.join(working_dir, 'test'), transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_ds = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_ds


def validate_and_test(
    batch_size: int,
    num_workers: int,
    drop_out: float,
    path_to_model_state: Path,
    path_to_dataset_dir : str = ''
) -> None:
    _, val_loader, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers, working_dir=path_to_dataset_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = EDMLoss().to(device)

    best_state = torch.load(path_to_model_state)

    model = create_model(drop_out=drop_out).to(device)
    model.load_state_dict(best_state["state_dict"])

    model.eval()
    
    val_loss = 0

    with torch.no_grad():
        for (x, y) in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            val_loss += criterion(p_target=y, p_estimate=y_pred).cpu()
            
    test_loss = 0
    with torch.no_grad():
        for (x, y) in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_loss += criterion(p_target=y, p_estimate=y_pred).cpu()
            
    logger.info(f"val loss {validate_losses.avg}; test loss {test_losses.avg}")


def get_optimizer(optimizer_type: str, model: NIMA, init_lr: float) -> torch.optim.Optimizer:
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.5, weight_decay=9)
    else:
        raise ValueError(f"not such optimizer {optimizer_type}")
    return optimizer


class Trainer:
    def __init__(
        self,
        *,
        num_epoch: int,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        experiment_dir: Path,
        drop_out: float,
        optimizer_type: str,
        dataset_dir : str = ''
    ):

        train_loader, val_loader, _ = get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            working_dir=dataset_dir
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = create_model(drop_out=drop_out)
        optimizer = get_optimizer(optimizer_type=optimizer_type, model=model, init_lr=init_lr)

        self.model = model

        pth = self.experiment_dir / "last_state.pth"
        if pth.exists():
            logger.info('load from last state')
            self.model.load_state_dict(torch.load(pth)['state_dict'])
        
        self.model = self.model.to(self.device)
        
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", patience=5)
        self.criterion = EDMLoss().to(self.device)
        
        # self.writer = SummaryWriter(str(experiment_dir / "logs"))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0
        self.print_freq = 100


    def train_model(self):
        best_loss = float("inf")
        best_state = None
        with tqdm(total=self.num_epoch * len(self.train_loader)) as pbar:
            for e in range(1, self.num_epoch + 1):
                train_loss = self.train(pbar)
                val_loss = self.validate()
                self.scheduler.step(metrics=val_loss)

                # self.writer.add_scalar("train/loss", train_loss, global_step=e)
                # self.writer.add_scalar("val/loss", val_loss, global_step=e)

                logger.info(f"current val loss: {val_loss}")

                if best_state is None or val_loss < best_loss:
                    logger.info(f"updated loss from {best_loss} to {val_loss}")
                    best_loss = val_loss
                    best_state = {
                        "state_dict": self.model.state_dict(),
                        "epoch": e,
                        "best_loss": best_loss,
                    }
                    torch.save(best_state, self.experiment_dir / "best_state.pth")
            torch.save({
                        "state_dict": self.model.state_dict(),
                        "epoch": e,
                        "best_loss": best_loss,
                    }, self.experiment_dir / "last_state.pth")

    def train(self, pbar):
        self.model.train()
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size
        train_loss = 0

        for (x, y) in self.train_loader:
            pbar.update(1)

            s = time.monotonic()

            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(p_target=y, p_estimate=y_pred)
            train_loss += loss.cpu()
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            
            # self.writer.add_scalar("train/current_loss", train_loss, self.global_train_step)
            self.global_train_step += 1

            e = time.monotonic()

        return train_loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                val_loss += self.criterion(p_target=y, p_estimate=y_pred).cpu()
                
                # self.writer.add_scalar("val/current_loss", val_loss, self.global_val_step)
                self.global_val_step += 1

        return val_loss
