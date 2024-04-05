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
                        A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_CONSTANT, value=0),
                        A.RandomCrop(width=32, height=32),
                        A.HorizontalFlip(p=0.5),
                        A.CoarseDropout (max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=0.4734, mask_fill_value = None),
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