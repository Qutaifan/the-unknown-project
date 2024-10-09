import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import logging
import pandas as pd
from model import SalesForecastNet
from data_processing import load_data, preprocess_data

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config['log_file'])
    logging.info("Starting training process.")
    
    # Load and preprocess data
    data_dir = 'Data/Monthly_sales'
    df = load_data(data_dir + '/1.xlsx')  # Example for 1.xlsx; modify as needed
    X = preprocess_data(df)
    y = df['Sales']  # Adjust based on your data
    
    # Convert to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # Initialize model, loss function, optimizer
    input_size = X.shape[1]
    model = SalesForecastNet(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), config['model_path'])
    logging.info("Training completed and model saved.")

if __name__ == '__main__':
    main()
