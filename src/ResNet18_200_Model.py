import os
import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18_pretrained():
    """
    加载预训练的 ResNet-18 模型
    """
    return models.resnet18(weights="IMAGENET1K_V1")


def get_resnet18():
    """
    创建一个 ResNet-18 模型
    """
    return models.resnet18()


def adjust_fc_out(resnet18: models.ResNet):
    """
    将 ResNet-18 模型的输出层大小设置为 200
    """
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 200)


class Model(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(Model, self).__init__()
        self.resnet18 = get_resnet18_pretrained() if pretrained else get_resnet18()
        adjust_fc_out(self.resnet18)

    def forward(self, x):
        return self.resnet18(x)

    def save(self, parent_dir):
        """
        保存模型参数
        :param parent_dir: 保存路径（文件夹）
        """
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        path = os.path.join(parent_dir, "ResNet18_200_Model.pth")
        torch.save(self.state_dict(), path)

    def load(self, parent_dir):
        """
        加载模型参数
        :param parent_dir: 保存路径（文件夹）
        """
        path = os.path.join(parent_dir, "ResNet18_200_Model.pth")
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
