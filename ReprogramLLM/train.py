from model import Model
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


configs = {
    "llm_layers": 5,
    "IT": {
        "in_features": 64,
        "out_features": 64
        
    },
    "OM": {
        "in_features": 10,
        "out_features": 10
        
    },
    
    "channels": [1, 32, 64]
    
}

model = Model(configs)
criterion = nn.CrossEntropyLoss()  # For classification tasks

# Dummy dataset
x_train = np.load("../test_data/X_4.npy")[:, np.newaxis, :, :]
y_train = np.load("../test_data/Y_4.npy") - 1

# Create a DataLoader
train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
train_loader = DataLoader(dataset=train_dataset, batch_size=16)

trained_parameters = []

for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in train_loader:
        
        # Forward pass
        outputs = model(inputs)
        
        # trained parameters and optimizer are assigned after one Forward pass, because:
        # linear layers are created after forward method is called once,
        # so the trained parameters are created after that.
        if len(trained_parameters) == 0:
            for p in model.parameters():
                if p.requires_grad is True:
                    trained_parameters.append(p)
            optimizer = torch.optim.SGD(trained_parameters, lr=0.01)
            
        optimizer.zero_grad()  # Zero the parameter gradients
        
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training')


