import time
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, criterion, optimizer, writer):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(device)

        curr_loss, curr_acc, n = 0.0, 0.0, 0
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs, 1)
            curr_loss += loss.item()
            curr_acc += (predicted == labels).sum().item()
            n += labels.size(0)

        avg_loss = curr_loss / n
        avg_acc = curr_acc / n
        self.writer.add_scalar('Train Loss', avg_loss, epoch)
        self.writer.add_scalar('Train Accuracy', avg_acc, epoch)

        return avg_loss, avg_acc

    def test_epoch(self, epoch, is_test=True):
        self.model.eval()
        self.model.to(device)

        curr_loss, curr_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            if not is_test:
                for i, (inputs, labels) in enumerate(self.valid_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    _, predicted = torch.max(outputs, 1)
                    curr_loss += loss.item()
                    curr_acc += (predicted == labels).sum().item()
                    n += labels.size(0)
            else:
                for i, (inputs, labels) in enumerate(self.test_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    _, predicted = torch.max(outputs, 1)
                    curr_loss += loss.item()
                    curr_acc += (predicted == labels).sum().item()
                    n += labels.size(0)

        avg_loss = curr_loss / n
        avg_acc = curr_acc / n
        if not is_test:
            self.writer.add_scalar('Valid Loss', avg_loss, epoch)
            self.writer.add_scalar('Valid Accuracy', avg_acc, epoch)

        return avg_loss, avg_acc

    def train(self, num_epochs, is_stg2=False):
        for epoch in range(num_epochs) if not is_stg2 else range(num_epochs, 2 * num_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            valid_loss, valid_acc = self.test_epoch(epoch, is_test=False)
            end_time = time.time()
            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
                f"Time: {end_time - start_time:.2f}s | "
            )

    def test(self):
        test_loss, test_acc = self.test_epoch(0, is_test=True)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
