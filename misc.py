import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder

class CustomTransforms:
    @staticmethod
    def transformA10():
        return A.Compose(
                    [
                        A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=(0.4914, 0.4822, 0.4465), always_apply=True),
                        A.RandomCrop(width=32, height=32),
                        A.HorizontalFlip(p=0.5),
                        A.CoarseDropout (max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=0.4734, mask_fill_value = None),
                        A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010),p =1.0),
                        ToTensorV2()
                    ]
                )

def max_lr_finder(model, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=1, num_iter=100, step_mode="linear")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()


class OneCyclePolicy:
    def __init__(self, max_lr, min_lr, epochs, steps_per_epoch, peak_epoch):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.peak_epoch = peak_epoch
        self.current_epoch = 0

    def get_lr(self):
        if self.current_epoch < self.peak_epoch:
            lr = self.min_lr + (self.max_lr - self.min_lr) * self.current_epoch / self.peak_epoch
        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * (self.current_epoch - self.peak_epoch) / (self.epochs - self.peak_epoch)
        return lr

    def step(self):
        self.current_epoch += 1
        return self.get_lr()