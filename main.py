import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from statistics import mean



device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda time: 
# device = 'cpu' # 
print(f"-- >> Using device {device} << --")

class SeqNet(nn.Module):
    def __init__(self, stack: list[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(*stack)
        # .to(device)

    def forward(self, x):
        return self.stack(x)

class RadioData(Dataset):
    def __init__(self, input_path:str, target_path:str, transform=None, target_transform=None) -> None:
        self.input_df = pd.read_csv(input_path, header=None)
        self.target_df = pd.read_csv(target_path, header=None)

        self.target_df.drop(columns=self.target_df.columns[-1],  axis=1,  inplace=True)

        self.input_df = self.input_df.to_numpy()
        self.target_df = self.target_df.to_numpy()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, id):
        train_data = self.input_df[id]
        train_ground_truth = self.target_df[id]

        train_data, train_ground_truth = torch.from_numpy(train_data), torch.from_numpy(train_ground_truth)

        if self.transform:
            # print(train_data)
            train_data = self.transform(train_data)

        if self.target_transform:
            train_ground_truth = self.target_transform(train_ground_truth)
        
        return train_data, train_ground_truth

def reset_weights(m):
  '''
    Reste parametrów wag
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train(model, dataset, lr, epochs_number: int, loss_function, optimizer, k_folds = 5, *args, **kwargs):
    start = time.time()
    print(f"Starting time: {start}")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))

    # model = model.to(device)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        
        print(f"Train dataset size: {len(train_ids)}")
        print(f"Test dataset size: {len(test_ids)}")
    
        # Print
        print(f' >> FOLD {fold + 1} <<')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        # batch_size size comment:
        # https://www.deeplearningbook.org/contents/optimization.html page 276
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=10, sampler=test_subsampler)
        
        # Init model
        model.apply(reset_weights)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses_per_epoch = []
        
        for epoch in tqdm(range(epochs_number), total=epochs_number, colour="GREEN", desc=f"Fold: {fold+1}/{k_folds}"):
            model.train()
            # print(f'Starting epoch {epoch+1}')

            current_loss = 0.0

            # Iterate over batches
            for data in trainloader:
                
                inputs, targets = data

                # inputs = inputs.to(device)
                # targets = targets.to(device)
                
                optimizer.zero_grad()
                
                outputs = model.forward(inputs)
                
                loss = loss_function(outputs, targets)
                
                loss.backward()
                
                optimizer.step()
                
                current_loss += loss.item()
            
            losses_per_epoch.append(current_loss)

        plt.clf()
        plt.plot(losses_per_epoch)
        plt.title(f"MSE - Fold 5")
    plt.savefig("./epochs_check.jpg")
    plt.clf()

    endtime = time.time()
    print(f"Endtime time: {endtime}")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(endtime)))
    print(f"Full training time [{device}]: {endtime - start}")

    path = f'./model-folds-{fold + 1}.pth'
    print(f'Training process has finished. Saving trained model in {path}.')
    save_path = path
    torch.save(model.state_dict(), save_path)

    # print('Starting testing')
    # Evaluation for this fold
    test_mse = []

    with torch.no_grad():
      model.eval()

      for _, data in enumerate(testloader, 0):

        inputs, targets = data

        outputs = model(inputs)

        test_mse.append(nn.MSELoss()(outputs, targets).item())

    print(f"Mean MSE from all {mean(test_mse)}")
    plt.plot(test_mse, color="g")
    plt.title(f"MSE on fold {fold+1} - test")
    plt.savefig(f"mse_fold_{fold+1}_model_seq_2.jpg")


if __name__ == "__main__":
    train_data = RadioData(
       "./data/radio_train/input_data.csv",
       "./data/radio_train/target_data.csv",
       transform=v2.ToDtype(torch.float32),
       target_transform=v2.ToDtype(torch.float32))

    model_seq = [
       nn.Linear(16, 8),
       nn.ReLU(),
       nn.Linear(8, 4),
       nn.ReLU(),
       nn.Linear(4, 2)
    ]

    model_seq_2 = [
       nn.Linear(16, 8),
       nn.Sigmoid(),
       nn.Linear(8, 4),
       nn.Sigmoid(),
       nn.Linear(4, 2)
    ]

    model_seq_3 = [
       nn.Linear(16, 32),
       nn.ReLU(),
       nn.Linear(32, 16),
       nn.ReLU(),
       nn.Linear(16, 4),
       nn.ReLU(),
       nn.Linear(4, 2)
    ]

    model_seq_4 = [
       nn.Linear(16, 4),
       nn.Sigmoid(),
       nn.Linear(4, 2)
    ]

    model_seq_test = [
       nn.Linear(16, 8),
       nn.ReLU(),
       nn.Linear(8, 4),
       nn.ReLU(),
       nn.Linear(4, 2)
    ]

    # network = SeqNet(model_seq_2) # Best
    network = SeqNet(model_seq_2)
    train(network, train_data, 1e-2, 100, nn.MSELoss(), None, k_folds = 5)
    # network = SeqNet(model_seq_test)
    # train(network, train_data, 1e-2, 500, nn.MSELoss(), None, k_folds = 5)
