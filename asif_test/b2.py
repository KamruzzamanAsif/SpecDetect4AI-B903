import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def fit(model, train_loader):
    """
    Train a classification model for 5 epochs using Adam optimizer and CrossEntropyLoss.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader with training batches (returns images and labels)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    num_epochs = 5
    total_batches = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()
            
            batch_count += 1
            total_batches += 1
            
            # Log progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batch_count
                accuracy = (epoch_correct / epoch_total) * 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Final epoch summary
        epoch_loss = epoch_loss / batch_count
        epoch_accuracy = (epoch_correct / epoch_total) * 100
        print(f"Epoch [{epoch+1}/{num_epochs}] - Final Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_accuracy:.2f}%\n")


