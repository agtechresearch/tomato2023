import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


from .log import regression_results

SEQUENCE_SIZE = 10
BATCH_SIZE = 32

class MyDataset:
    def __init__(self, df, x_cols=None, y_cols=None):    
        self.df = df
        self.x_cols = x_cols if x_cols else list(df.columns)
        self.y_cols = y_cols if y_cols else list(df.columns)
        self.x_train_no_window = None
        self.x_test_no_window = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.train_loader = None
        self.test_loader = None
    
    def print_loader(self):
        for loader in [self.train_loader, self.test_loader]:
            if loader:
                print(f"Total data: {len(loader)}")
                for t in loader:
                    print(f"train: {t[0].shape}, test: {t[1].shape}")
            else:
                print("Loader is not initialized. Run preprocessing() function.")

    def _to_sequences(self, df, seq_size=SEQUENCE_SIZE):
        num_len = df.shape[0] - seq_size
        x = np.zeros((num_len, seq_size, len(self.x_cols)))
        y = np.zeros((num_len, len(self.y_cols)))
        df_x = df[self.x_cols]
        df_y = df[self.y_cols]
        for i in range(len(df) - seq_size):
            x[i] = df_x.iloc[i:(i + seq_size)].to_numpy()
            y[i] = df_y.iloc[i + seq_size -1].to_numpy() 
            # -1 하면 그 날짜, 안하면 다음날 예측
        return (
            torch.tensor(x, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.float32) 
        )

    def _train_test_split(self, train_ratio=0.8):
        train_size = int(self.df.shape[0] * train_ratio) 
        df_train = self.df.iloc[:train_size]
        df_test = self.df.iloc[train_size:]
        x_train, y_train = self._to_sequences(df_train)
        x_test, y_test = self._to_sequences(df_test)
        return x_train, y_train, x_test, y_test
        

    def preprocessing(self, train_ratio=0.8):
        x_train, y_train, x_test, y_test = self._train_test_split(train_ratio)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, drop_last=True)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                shuffle=False, drop_last=True)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.x_train_no_window = x_train[:,-1,:]
        self.x_test_no_window = x_test[:,-1,:]
        left_train = self.x_train_no_window.shape[0] - self.x_train_no_window.shape[0]%BATCH_SIZE
        left_test = self.x_test_no_window.shape[0] - self.x_test_no_window.shape[0]%BATCH_SIZE
        self.x_train_no_window = self.x_train_no_window[:left_train, :]
        self.x_test_no_window = self.x_test_no_window[:left_test, :]
        
        return train_loader, test_loader
    
class Modeling:
    criterion = { # https://nuguziii.github.io/dev/dev-002/
        "mse": nn.MSELoss,
        "bce": nn.BCELoss,
        "bceSigmoid": nn.BCEWithLogitsLoss, # 이미 sigmoid 있음
        "crossEntropy": nn.CrossEntropyLoss, # 이미 sigmoid 있음
    }
    optimizer = {
        "SparseAdam": torch.optim.SparseAdam,
        "Adam": torch.optim.Adam, 
    }

    def __init__(self, model, data: MyDataset, 
                 taskType="regression", criterion="mse",
                 lr=0.001, device=None):
        
        if not device:
            has_mps = torch.backends.mps.is_built()
            device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
        self.device = device

        self.model = model(
            input_dim=len(data.x_cols), 
            output_dim=len(data.y_cols),
            device=device,
            # taskType=taskType
        ).to(device)

        # if "classification" in taskType:
        #     criterion = "crossEntropy"
        self.criterion = Modeling.criterion[criterion]()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', factor=0.5, patience=3, verbose=True)


    def train(self, epochs, train_loader, test_loader, early_stop_count=None):
        # min_val_loss = float('inf')
        history = {
            "epoch": [], 
            "train_loss": [],
            "val_loss": []            
        }
        for epoch in range(1, epochs+1):
            self.model.train()
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = \
                    x_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch) 
                train_loss = self.criterion(outputs, y_batch)
                train_loss.backward()
                self.optimizer.step()

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in test_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = \
                        x_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(x_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            # self.scheduler.step(val_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Traini Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss.item())
                history["val_loss"].append(val_loss)

            # if early_stop_count:
            #     if val_loss < min_val_loss:
            #         min_val_loss = val_loss
            #         early_stop_count = 0
            #     else:
            #         early_stop_count += 1

            #     if early_stop_count >= 5:
            #         print("Early stopping!")
            #         break
        return history

    def eval(self, loader, y):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                x_batch, _ = batch
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                predictions.extend(outputs.squeeze().tolist())

        y_true = y[:len(predictions)].numpy()
        y_pred = np.array(predictions)        

        # regression_results(
        #     y_true.reshape(-1, 1),
        #     y_pred.reshape(-1, 1)
        # )

        return y_true, y_pred
