import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
from torch_lr_finder import LRFinder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

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

def plot_gradcam(images, target_layers):
    num_images = len(images)
    for i in range(num_images):
        # Convert image to tensor and move to device
        img_tensor = images[i].unsqueeze(0).to(device)
        # Normalize image tensor
        img_tensor_normalized = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        # Forward pass
        num_plots = len(target_layers)+1
        plt.figure(figsize=(10, 5))
        plt.subplot(1, num_plots, 1)
        plt.imshow(np.transpose(images[i]/ 2 + 0.5, (1, 2, 0)))
        # plt.title(f'(Actual: {classes[correct_labels[i]]}, Predicted: {classes[misclassified_labels[i]]})')
        plt.axis('off')
        for i, layer in enumerate(target_layers):
            gradcam = GradCAM(model=model, target_layers=[layer])
            out = gradcam.forward(img_tensor_normalized, None, eigen_smooth=True)
            img_with_heatmap = show_cam_on_image(np.transpose(img_tensor_normalized.squeeze().cpu().numpy(), (1, 2, 0)), out.squeeze(), use_rgb=True)
            plt.subplot(1, num_plots, i+2)
            plt.imshow(img_with_heatmap)
            # plt.title('GradCAM')
            plt.axis('off')
        plt.show() 

def get_misclassified(model, device, test_loader, n=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for i, (img, x,y) in enumerate(zip(data, pred, target)):
                if(x != y):
                    misclassified_images.append(img.cpu())
                    misclassified_labels.append(x.cpu())
                    correct_labels.append(y.cpu())
                if len(misclassified_images) >= n:
                    break
                    
                
            break
        return misclassified_images, misclassified_labels, correct_labels
