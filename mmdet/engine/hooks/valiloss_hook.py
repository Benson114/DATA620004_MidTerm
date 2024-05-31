from mmdet.apis import init_detector, single_gpu_test
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter

@HOOKS.register_module()
class ComputeValMetrics(Hook):
    def __init__(self, interval: int = 1, log_dir=None) -> None:
        super().__init__()
        self.interval = interval
        self.writer = SummaryWriter(log_dir="/root/model_saved/neuralnetwork_hw2/mmdetection/mmdetection/tensorboard_vali")  # 初始化 TensorBoard 的写入器

    def after_val(self, runner, **kwargs):
        # 每隔一定间隔执行
        if runner.epoch % self.interval == 0:
            checkpoint_file = runner.work_dir + f"/epoch_{runner.epoch}.pth"
            model = init_detector(runner.cfg, checkpoint_file, device='cuda:0')

            # 验证集的 DataLoader
            data_loader = runner.data_loader['val']

            # 使用单 GPU 测试，计算验证结果
            outputs = single_gpu_test(model, data_loader, show=False)

            # 评估模型性能
            eval_results = runner.evaluate(outputs, runner.cfg.data.val)
            mAP = eval_results['mAP']

            # 可以添加记录损失的逻辑，如果损失不是由 evaluate 返回，需要额外计算
            # 假设 evaluate 返回了损失
            val_loss = eval_results.get('loss', np.nan)

            # 记录到 TensorBoard
            self.writer.add_scalar('Validation/mAP', mAP, runner.epoch)
            self.writer.add_scalar('Validation/Loss', val_loss, runner.epoch)

    def after_run(self, runner, **kwargs):
        # 运行结束后关闭 TensorBoard 的写入器
        self.writer.close()
