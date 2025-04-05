import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'dataset'  # Expected structure: dataset/train/, dataset/val/

# === Load pretrained model and preprocessing transforms ===
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
preprocess = weights.transforms()

# === Datasets and DataLoaders ===
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=preprocess)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# === Modify the final layer for our number of classes ===
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

# === Loss function, optimizer, and learning rate scheduler ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# === Training loop ===
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    print(f"📚 Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"✅ Validation Accuracy: {val_accuracy:.2f}%")

    scheduler.step()

# === Save model state dict ===
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/model_state_dict.pt')
print("✅ Model state dict saved to model/model_state_dict.pt")
