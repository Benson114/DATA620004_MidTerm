import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.CUB200Dataset import train_dataset, valid_dataset, test_dataset
from src.ResNet18_200_Model import Model
from src.ResNet18_200_Trainer import Trainer
from src.config import *


def main():
    writer = SummaryWriter('runs/CUB200_train_joint')  # Initialize TensorBoard

    train_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=dataloader_params["batch_size"] * n_gpus,
        num_workers=dataloader_params["num_workers"] * n_gpus,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, shuffle=False,
        batch_size=dataloader_params["batch_size"] * n_gpus,
        num_workers=dataloader_params["num_workers"] * n_gpus,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False,
        batch_size=dataloader_params["batch_size"] * n_gpus,
        num_workers=dataloader_params["num_workers"] * n_gpus,
        pin_memory=True
    )

    model = Model(pretrained=True)
    criterion = CrossEntropyLoss()
    optim_params = [
        {
            "params": model.resnet18.fc.parameters(),
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.001,
        },
        {
            "params": (param for name, param in model.named_parameters() if "fc" not in name),
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.001,
        },
    ]
    optimizer = optim.SGD(optim_params)

    trainer = Trainer(model, train_loader, valid_loader, test_loader, criterion, optimizer, writer)
    trainer.train(num_epochs * 2)
    trainer.test()

    model.save("models/train_joint")

    writer.close()  # Close the TensorBoard writer
    return model


if __name__ == "__main__":
    main()
