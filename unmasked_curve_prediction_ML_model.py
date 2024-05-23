import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the datasets from the pickle files
with open('masked_curves.pkl', 'rb') as f:
    masked_curves = pickle.load(f)

with open('unmasked_curves.pkl', 'rb') as f:
    unmasked_curves = pickle.load(f)

# Convert lists to numpy arrays
masked_curves = np.array(masked_curves)
unmasked_curves = np.array(unmasked_curves)

# Flatten the curves for normalization
num_curves, num_points, num_features = masked_curves.shape
masked_curves_flat = masked_curves.reshape((num_curves, -1))
unmasked_curves_flat = unmasked_curves.reshape((num_curves, -1))

# Normalize the curves
scaler = MinMaxScaler()
masked_curves_flat = scaler.fit_transform(masked_curves_flat)
unmasked_curves_flat = scaler.transform(unmasked_curves_flat)

# Reshape back to original shape
masked_curves = masked_curves_flat.reshape((num_curves, num_points, num_features))
unmasked_curves = unmasked_curves_flat.reshape((num_curves, num_points, num_features))

# Handle NaN values by replacing them with zeros
masked_curves = np.nan_to_num(masked_curves, nan=0.0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(masked_curves, unmasked_curves, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

input_size = num_features
hidden_size = 128
num_layers = 2
output_size = num_features

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2
clip_value = 1.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
print(f'Test Loss: {test_loss / len(test_loader):.4f}')

model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

# Concatenate all predictions
predictions = np.concatenate(predictions, axis=0)

# Optionally, inverse transform the predictions to the original scale
predictions_flat = predictions.reshape((predictions.shape[0], -1))
predictions_flat = scaler.inverse_transform(predictions_flat)
predictions = predictions_flat.reshape((predictions.shape[0], num_points, num_features))

# Function to plot actual and predicted curves superimposed
def plot_predictions(masked, predicted, actual, num_samples=5):
    fig = plt.figure(figsize=(15, num_samples * 5))
    
    for i in range(num_samples):
        idx = np.random.randint(0, masked.shape[0])

        ax = fig.add_subplot(num_samples, 1, i + 1, projection='3d')
        ax.plot(predicted[idx, :, 0], predicted[idx, :, 1], predicted[idx, :, 2], label='Predicted', color='orange')
        ax.plot(actual[idx, :, 0], actual[idx, :, 1], actual[idx, :, 2], label='Actual', color='green')
        ax.set_title(f'Curve {idx + 1}')
        ax.legend()
    
    plt.show()

# Visualize some sample predictions
plot_predictions(X_test.numpy(), predictions, y_test.numpy(), num_samples=2)
