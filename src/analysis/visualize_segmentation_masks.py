import matplotlib.pyplot as plt

from src.pipeline.data.data_loader import get_data_loader
from src.utils.project_directories import get_data_subdir_str


def visualize_segmentation_masks(data_loader):
    images, masks = next(iter(data_loader))

    fig, ax = plt.subplots(nrows=2, ncols=len(images), figsize=(15, 10))
    for i in range(len(images)):
        ax[0, i].imshow(images[i].permute(1, 2, 0))
        ax[0, i].set_title('Original Image')
        ax[0, i].axis('off')

        visible_mask = masks[i].squeeze() * 255  # Scale mask for visibility
        ax[1, i].imshow(visible_mask, cmap='gray')
        ax[1, i].set_title('Segmentation Mask')
        ax[1, i].axis('off')

    plt.show()


if __name__ == '__main__':
    root_dir = get_data_subdir_str()

    data_loader = get_data_loader(root_dir)
    visualize_segmentation_masks(data_loader)

