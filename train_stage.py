import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.CUB200Dataset import train_dataset, valid_dataset, test_dataset
from src.ResNet18_200_Model import Model
from src.ResNet18_200_Trainer import Trainer
from src.config import *


def main():
    writer = SummaryWriter('runs/CUB200_train_stage')  # Initialize TensorBoard

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

    # Stage 1: Training only the new fully connected layer
    for param in model.resnet18.parameters():
        param.requires_grad = False
    for param in model.resnet18.fc.parameters():
        param.requires_grad = True

    optim_params_stg1 = {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.001,
    }
    optimizer_stg1 = optim.SGD(model.resnet18.fc.parameters(), **optim_params_stg1)

    trainer_stg1 = Trainer(model, train_loader, valid_loader, test_loader, criterion, optimizer_stg1, writer)
    trainer_stg1.train(num_epochs, is_stg2=False)
    trainer_stg1.test()

    # Stage 2: Fine-tuning the whole network
    for param in model.parameters():
        param.requires_grad = True

    optim_params_stg2 = {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.001,
    }
    optimizer_stg2 = optim.SGD(model.parameters(), **optim_params_stg2)

    trainer_stg2 = Trainer(model, train_loader, valid_loader, test_loader, criterion, optimizer_stg2, writer)
    trainer_stg2.train(num_epochs, is_stg2=True)
    trainer_stg2.test()

    model.save("models/train_stage")

    writer.close()  # Close the TensorBoard writer
    return model


if __name__ == "__main__":
    main()
