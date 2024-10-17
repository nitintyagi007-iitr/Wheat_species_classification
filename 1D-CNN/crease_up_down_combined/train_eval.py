import torch
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Train_Eval:
    def __init__(self, model, model_name, device, train_loader=None, val_loader=None, optimizer = None, criterion = None,  lr_scheduler= None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.total_steps_tr = len(self.train_loader)
        self.total_steps_val = len(self.val_loader)
        self.valid_loss_min = np.Inf
        self.lr_scheduler = lr_scheduler
        self.model_name = model_name

    def __train__(self, epoch):
        self.model.train()
        train_acc = 0
        train_loss = 0
        pbar = tqdm(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets.long())
            self.optimizer.zero_grad()    # clear the gradients of all optimized variables
            loss.backward()               # back_prop: compute gradient of the loss with respect to model parameters
            self.optimizer.step()         # parameter update
            train_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct = torch.sum(pred==targets).item()
            total = targets.size(0)
            train_acc += correct/total

        tr_loss = train_loss/self.total_steps_tr
        tr_acc = train_acc/self.total_steps_tr
        return tr_loss, tr_acc

    def __validation__(self, epoch):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        pbar = tqdm(self.val_loader)

        with torch.no_grad():           # disable gradient calculation
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.long())

                val_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                correct = torch.sum(pred==targets).item()
                total = targets.size(0)
                val_acc += correct/total

            val_loss = val_loss/self.total_steps_val
            val_acc = val_acc/self.total_steps_val
        return val_loss, val_acc

    def run(self, epochs)-> dict:
        train_losses = []
        train_acc = []
        test_losses = []
        test_acc = []
        for epoch in range(epochs):
            print(f'\nEpoch: {epoch}')
            train_epoch_loss, train_epoch_acc = self.__train__(epoch)
            test_epoch_loss, test_epoch_acc = self.__validation__(epoch)
            
            
            
            print ('Epoch [{}] --> LossTr: {:.4f}    AccTr: {:.4f}'
                    .format(epoch, train_epoch_loss, train_epoch_acc), end = '    ')
            print('lossVal : {:.4f}     accVal : {:.4f}\n'.format(test_epoch_loss , test_epoch_acc))

            train_losses.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            test_losses.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)

            self.lr_scheduler.step(train_losses[-1])   # updating learning rate

            network_learned = self.valid_loss_min - test_losses[-1] > 0.01
            if network_learned:
                self.valid_loss_min = test_losses[-1]
                torch.save(self.model.state_dict(), '{name}_model.pt'.format(name = self.model_name))
                print('Detected network improvement, saving current model  "\u2705"')

        history = {
            'train_loss': train_losses,
            'train_acc': train_acc,
            'val_loss': test_losses,
            'val_acc': test_acc
        }

        with open('{x}_history.pickle'.format(x = self.model_name), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        return history
    
    def eval(self , df_tst, mean ,std):
        self.model.eval()
        X = torch.tensor(df_tst.iloc[:, :-1].values, dtype=torch.float32)
        Y = torch.tensor( df_tst.iloc[:, -1].values, dtype=torch.float32)

        X = (X -mean) / std  # apply snv
        output = self.model(X.to(self.device))
        target =  torch.Tensor(Y).to(self.device)
        total = len(df_tst)
        _,pred = torch.max(output, dim=1)
        correct = torch.sum(pred==target).item()
        print('acc : {:4f}   '.format((correct/total)  ))
        t = target.cpu().numpy()
        dict_ = {}
        dict_[0] , dict_[1] ,dict_[2] ,dict_[3]= 0 ,0 ,0 ,0
        for x in t:
            dict_[x] +=1
        print(dict_)
        
        dict_ = {
            'y_pred' : pred.cpu().numpy(),
            'y_true' : target.cpu().numpy()
        }
        with open('{x}_prediction.pickle'.format(x = self.model_name), 'wb') as handle:
            pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
        
        return

    
        

    
      
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MyDataset(Dataset):
  # defining values in the constructor
  def __init__(self, path, mean, std ,apply_transform:bool = False):
      self.df = pd.read_csv(path)

      self.X = torch.tensor(self.df.iloc[:, :-1].values, dtype=torch.float32)
      self.Y = torch.tensor( self.df.iloc[:, -1].values, dtype=torch.float32)
      self.shape = self.df.shape
      self.mean = mean
      self.std = std
      self.apply_transform = apply_transform
    
  # Getting the data samples
  def __getitem__(self, idx):
      sample = self.X[idx], self.Y[idx]
      
      if self.apply_transform:
          x = self.__snv__(self.X[idx])
          sample = x, self.Y[idx] 
      return sample
  
  def __len__(self):
      return self.shape[0]
  
  def __snv__(self ,input_data):

    # Define a new array and populate it with the corrected data 
    return (input_data - self.mean) /( self.std)
             