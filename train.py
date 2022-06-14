import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch import optim


from train import matches, transforms, models



def train_epoch(model: nn.Module, train_dataset, val_dataset, opt, loss_fn):
    model.train()
    num_correct = 0
    num_seen = 0
    for data in train_dataset:
        X = data["game"]
        y_true = data["team1Won"]
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        opt.step()
        num_correct += (y_true == (y_pred >= 0.5)).float().sum()
        num_seen += len(y_true)
        accuracy = num_correct / num_seen
        print(f"\r[Train] Loss: {loss:.4f} | Acc: {accuracy * 100:.2f}%", end="")
    print()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        num_correct = 0
        num_seen = 0
        for data in val_dataset:
            X = data["game"]
            y_true = data["team1Won"]
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)
            val_loss = loss
            num_correct += (y_true == (y_pred >= 0.5)).float().sum()
            num_seen += len(y_true)
            accuracy = num_correct / num_seen
            print(f"\r[Val] Loss: {loss:.4f} | Acc: {accuracy * 100:.2f}%", end="")
        print()
    return val_loss


def main():
    device = torch.device("cpu")
    dataset = matches.MatchesDataset(
        "./data", transforms=[transforms.ToTensor()]
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
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.1, patience=15)
    for i in range(1, 150 + 1):
        print(f"Epoch {i}")
        val_loss = train_epoch(model, train_dataset, val_dataset, opt, loss_fn)
        scheduler.step(val_loss)
    torch.save(model.state_dict(), "models/model_1.pt")


if __name__ == "__main__":
    main()
