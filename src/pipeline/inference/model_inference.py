import torch


# Inference Function to process the entire dataset
def infer(model, data_loader, device):
    model.eval()
    all_images = []
    all_true_masks = []
    all_pred_masks = []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            all_images.append(images.cpu())
            all_true_masks.append(masks.cpu())
            all_pred_masks.append(outputs.cpu())

    return torch.cat(all_images), torch.cat(all_true_masks), torch.cat(all_pred_masks)
