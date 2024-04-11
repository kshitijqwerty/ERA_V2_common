import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
from torch_lr_finder import LRFinder

def transform_cifar_A10():
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

def transform_cifar_A11():
    return A.Compose(
        [
            A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=(0.4914, 0.4822, 0.4465), always_apply=True),
            A.RandomCrop(width=32, height=32),
            A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=0.4734, mask_fill_value = None),
            A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010),p =1.0),
            ToTensorV2()
        ]
    )

def transform_cifar_test():
    return A.Compose(
        [
            A.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), p=1.0
            ),
            ToTensorV2(),
        ]
    )

def max_lr_finder(model, train_loader, optimizer, criterion, device):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()