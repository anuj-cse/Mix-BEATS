import numpy as np
import pandas as pd

import os
import sys
sys.path.append('./model')

import torch
from torch.utils.data import Dataset, DataLoader
from model.nbeats import NBeatsNet
from torch.utils.data import ConcatDataset

from tqdm import tqdm
from time import time
from sklearn.metrics import mean_squared_error


def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std+eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    return predictions * (std+eps) + mean


class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length, stride=1):
        # Standardize the time series data
        self.data, self.mean, self.std = standardize_series(data)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.backcast_length - self.forecast_length) // self.stride + 1

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.backcast_length]
        y = self.data[start_index + self.backcast_length : start_index + self.backcast_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_datasets(folder_path, backcast_length, forecast_length, stride):
    datasets = []

    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)
        for building in os.listdir(region_path):

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                dataset = TimeSeriesDataset(energy_data, backcast_length, forecast_length, stride)
                datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


def train(model, criterion, optimizer, device, train_loader, val_loader):

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    early_stop = False

    num_epochs = 100
    train_start_time = time()  # Start timer

    for epoch in range(num_epochs):

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break  

        model.train()
        train_losses = []

        epoch_start_time = time()  # Start epoch timer

        # Progress bar for the training loop
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for x_batch, y_batch in pbar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                backcast, forecast = model(x_batch)
                loss = criterion(forecast, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                pbar.set_postfix(loss=loss.item(), elapsed=f"{time() - epoch_start_time:.2f}s")
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        y_true_val = []
        y_pred_val = []

        # Progress bar for the validation loop
        with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for x_val, y_val in pbar:
                x_val, y_val = x_val.to(device), y_val.to(device)
                with torch.no_grad():
                    backcast, forecast = model(x_val)
                    loss = criterion(forecast, y_val)
                    val_losses.append(loss.item())
                    
                    # Collect true and predicted values for RMSE calculation
                    y_true_val.extend(y_val.cpu().numpy())
                    y_pred_val.extend(forecast.cpu().numpy())

        # Calculate average validation loss and RMSE
        avg_val_loss = np.mean(val_losses)
        rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save the best model parameters
            torch.save(model.state_dict(), './model_weights/nbeats_1024/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                early_stop = True


    total_training_time = time() - train_start_time
    print(f'Total Training Time: {total_training_time:.2f}s')



if __name__ == '__main__':

    
    # Parameters
    backcast_length = 168
    forecast_length = 24
    stride = 24
    batch_size = 1024

    # Load datasets
    train_datasets = load_datasets('./dataset/train', backcast_length, forecast_length, stride)
    val_datasets = load_datasets('./dataset/val', backcast_length, forecast_length, stride)

    # Create data loaders
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

    patch_size = 8 
    num_patches = backcast_length // patch_size

    # check device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define N-BEATS model
    model = NBeatsNet(
        device=device,
        forecast_length=forecast_length,
        backcast_length=backcast_length
    ).to(device)

    # model's parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model's parameter count is:", param)



    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training the model and save best parameters
    train(model=model, criterion=criterion, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader)
