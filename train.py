import argparse
import time

import message_center as mc

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch import optim
from torch.nn import functional as F

from train import matches, transforms, models


service = mc.Service("lolstats")


def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        res = {}
        for key in x.keys():
            res[key] = move_to_device(x[key], device)
        return res
    elif isinstance(x, list):
        return [move_to_device(e, device) for e in x]
    raise ValueError(f"Unsupported class {type(x)}")


def train(model: nn.Module, train_dataset, opt, loss_fn, device):
    model.train()
    num_correct = 0
    num_seen = 0
    msg_id = None
    for i, data in enumerate(train_dataset):
        start_time = time.time()
        X = move_to_device(data["game"], device)
        y_true = data["team1Won"].to(device)
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt.step()
        loss = loss.item()
        num_correct += (y_true == (y_pred >= 0.5)).float().sum()
        num_seen += len(y_true)
        accuracy = num_correct / num_seen
        elapsed_time = time.time() - start_time
        msg = f"[Train] (Time per batch: {elapsed_time:.3f} s) ({i+1} / {len(train_dataset)}) Loss: {loss:.4f} | Acc: {accuracy * 100:.2f}%"
        msg_id = service.add_message(msg, message_id=msg_id)
        print(f"\r{msg}", end="")
    print()


def validate(model: nn.Module, val_dataset, loss_fn, device):
    model.eval()
    val_loss = 0
    msg_id = None
    with torch.no_grad():
        num_correct = 0
        num_seen = 0
        for i, data in enumerate(val_dataset):
            X = move_to_device(data["game"], device)
            y_true = move_to_device(data["team1Won"], device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)
            val_loss = loss.item()
            num_correct += (y_true == (y_pred >= 0.5)).float().sum()
            num_seen += len(y_true)
            accuracy = num_correct / num_seen
            msg = f"[Val] ({i+1} / {len(val_dataset)}) Loss: {val_loss:.4f} | Acc: {accuracy * 100:.2f}%"
            msg_id = service.add_message(msg, message_id=msg_id)
            print(f"\r{msg}", end="")
        print()
    return val_loss


def main(device: str, batch_size: int):
    device = torch.device(device)
    dataset = matches.MatchesDataset(
        "./data", sample_transforms=[transforms.TeamShuffle(), transforms.ToTensor()]
    )
    val_split = 0.3
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, lengths=[train_size, val_size])
    print(f"Training on {train_size:,} samples")
    print(f"Validating on {val_size:,} samples")
    train_dataset = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataset = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model = models.MatchModel()
    model = model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", factor=0.1, patience=15
    )
    for i in range(10):
        print(f"Warmup {i+1} / 10")
        for p in opt.param_groups:
            p["lr"] = (i + 1) / 10 * 1e-3
        train(model, train_dataset, opt, loss_fn, device)
    for i in range(1, 150 + 1):
        print(f"Epoch {i} / 150")
        train(model, train_dataset, opt, loss_fn, device)
        val_loss = validate(model, val_dataset, loss_fn, device)
        scheduler.step(val_loss)
        torch.save(model.state_dict(), "models/model_1.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    main(device=args.device, batch_size=args.batch_size)
