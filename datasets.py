from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


# Custom Class to work with albumentations lib
class CIFAR10Custom(datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None, batch_size=128, use_cuda=False):
        self.dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=64)
        if(transform is None and train):
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                    A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=0.4734, mask_fill_value = None),
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010),p =1.0),
                    ToTensorV2()
                ]
            )
        elif(transform is None):
            transform = A.Compose(
                [
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010),p =1.0),
                    ToTensorV2()
                ]
            )
        self.transform = transform
        super().__init__(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, index):
        image, lab = self.data[index], self.targets[index]
        if(self.transform is not None):
            image = self.transform(image=image)
        return image['image'], lab
    
    def get_dataloader(self):
        return DataLoader(self, **self.dataloader_args)