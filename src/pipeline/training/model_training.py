import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def train_model(model, data_loader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Set the model to training mode
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Finished Training')

    # Plot training loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o', color='b')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    return epoch_losses
