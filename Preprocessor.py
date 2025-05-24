
from torchvision import transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class CTPreprocessor:
    
    def __init__(
            self,
            img_size:tuple = None,
            transformations: list = [],
            use_mask = False
    ) -> None:
        
        transform_list = [transforms.Resize([224, 224] if img_size is None else img_size)]
        for trans in transformations:
            transform_list.append(trans)
        self.transform = transforms.Compose(transform_list)
        self.use_mask = use_mask

    def _imshow(self, img, title = "temp_img"):
        if isinstance(img, torch.Tensor):  # Check if `img` is a PyTorch tensor
            npimg = np.transpose(img.numpy(), (1, 2, 0))
        elif isinstance(img, np.ndarray):
            npimg = img
        plt.imshow(npimg, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()


    def _getmask(self, image, show_all = False, show_last = False):
        if(isinstance(image, torch.Tensor)):
            image_np = (np.transpose(image.detach().numpy(), (1, 2, 0)))
        elif(isinstance(image, Image.Image)):
            image_np = np.array(image)
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)

        # Binarize the image
        _, binary_mask = cv2.threshold(image_np, 100, 255, cv2.THRESH_BINARY)

        # Erode the binary mask
        kernel = np.ones((2, 2), np.uint8)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=3)
        blurred_mask = cv2.blur(eroded_mask, (21,21))
        
        # Find contours
        contours, _ = cv2.findContours(blurred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = cv2.drawContours(blurred_mask.copy(), contours, -1, (255), thickness=cv2.FILLED)


        # Flood fill
        flood_filled_mask = contour_mask.copy()
        height, width = flood_filled_mask.shape
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(flood_filled_mask, mask, seedPoint=(0, 0), newVal=0)

        # Convert to PyTorch tensor
        mask_tensor = transforms.ToTensor()(Image.fromarray(flood_filled_mask))

        if (show_all):
            self._imshow(binary_mask, "THRESHOLDING")
            self._imshow(eroded_mask, "EROSION")
            self._imshow(blurred_mask, "BLURRING")
            # imshow(dilated_mask, "DILATION")
            self._imshow(contour_mask, "CONTOUR")
            self._imshow(flood_filled_mask, "FLODD FILL")
            self._imshow(mask_tensor, "FINAL")
        elif(show_last):
            self._imshow(mask_tensor, "FINAL")
        
        return mask_tensor / 255.0

    def __call__(self, img):
        if(self.use_mask):
            trans_img = self.transform(img)
            mask = self._getmask(trans_img)
            return trans_img * mask # Elementwise multiplication
        else:
            return self.transform(img)

        
if __name__ == "__main__":
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    def imshow(img, title):
            # img = img / 2 + 0.5  # Unnormalize
            # axes, fig = plt.figure(num = (1,4), figsize=(5,12)) 
            npimg = img.numpy()
            # plt.title()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.title(title)
            plt.show()


    transform = CTPreprocessor((256,256))
    dataset = ImageFolder(root="./Data/Compiled/PNG/", transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Get a batch of training data
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # Show images
    imshow(make_grid(images[labels == 0][:3], padding=20, pad_value=1), dataset.classes[0])
    # imshow(make_grid(images[labels == 1][:3], padding=20, pad_value=1), dataset.classes[1])
    # imshow(make_grid(images[labels == 2][:3], padding=20, pad_value=1), dataset.classes[2])
    
