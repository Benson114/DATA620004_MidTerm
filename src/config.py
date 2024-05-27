import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"

import random
import torch

n_gpus = torch.cuda.device_count()

root_dir = "data/CUB_200_2011"
dataloader_params = {
    "batch_size": 32,
    "num_workers": 4,
    "valid_rate": 0.1,
}
num_epochs = 50

seed = 42
random.seed(seed)
torch.manual_seed(seed)
