import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch import optim


from train import matches, transforms, models


def train_epoch(model: nn.Module, train_dataset, val_dataset, opt, loss_fn, device):
    model.train()
    num_correct = 0
    num_seen = 0
    for data in train_dataset:
        X = data["game"].to(device)
        y_true = data["team1Won"].to(device)
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt.step()
        num_correct += (y_true == (y_pred >= 0.5)).float().sum()
        num_seen += len(X)
        accuracy = num_correct / num_seen
        print(f"\r[Train] Loss: {loss:.4f} | Acc: {accuracy * 100:.2f}%", end="")
    print()

    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_seen = 0
        for data in val_dataset:
            X = data["game"].to(device)
            y_true = data["team1Won"].to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)
            num_correct += (y_true == (y_pred >= 0.5)).float().sum()
            num_seen += len(X)
            accuracy = num_correct / num_seen
            print(f"\r[Val] Loss: {loss:.4f} | Acc: {accuracy * 100:.2f}%", end="")
        print()


def main():
    device = torch.device("mps")
    dataset = matches.MatchesDataset(
        "./data", transforms=[transforms.TeamShuffle(), transforms.ToTensor()]
    )
    val_split = 0.3
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, lengths=[train_size, val_size])
    train_dataset = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
    )
    val_dataset = DataLoader(
        dataset=val_dataset,
        batch_size=128,
        shuffle=True,
    )

    model = models.MatchModel()
    model = model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for i in range(1, 10 + 1):
        print(f"Epoch {i}")
        train_epoch(model, train_dataset, val_dataset, opt, loss_fn, device)


if __name__ == "__main__":
    main()
