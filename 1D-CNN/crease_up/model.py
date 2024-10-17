import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, drop ,in_size, out_size):
      super(Dense ,self).__init__()
      self.dropout = nn.Dropout(drop)
      self.linear = nn.Linear(in_size, out_size)
      self.prelu = nn.PReLU()

    def forward(self, x):
      x = self.dropout(x)
      x = self.linear(x)
      x = self.prelu(x)
      return x
    

class Conv_1D(nn.Module):
  def __init__(self , in_channels, out_channels, kernel_size=3, stride = 1):
    super(Conv_1D, self).__init__()
    self.batch = nn.BatchNorm1d(in_channels)
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size , stride = stride)
    self.prelu = nn.PReLU()
    self.pool = nn.MaxPool1d(2)

  def forward(self, x):
    x = self.batch(x)
    x = self.conv(x)
    x = self.prelu(x)
    return self.pool(x)
  

class CNN1D(nn.Module):
    def __init__(self, num_classes, use_svm = False):
      super(CNN1D ,self).__init__()
      self.c1 = Conv_1D(1,32 ,3,1)
      self.c2 = Conv_1D(32, 64 , 5, stride=2)
      self.c3 = Conv_1D(64, 128, 5, stride=2)
      self.use_svm = use_svm
      self.flatten = nn.Flatten(start_dim = 1)
      self.d2= Dense(0.25 , 384 ,256)
      self.d3 = Dense(0.1 , 256 , 64)
      self.d4 = Dense(0,64 , num_classes)


      self.acti = nn.PReLU()
      self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

      x = torch.unsqueeze(x, 1)
      c1 = self.c1(x)

      c2 = self.c2(c1)
      c3 = self.c3(c2)
      c = self.flatten(c3)
      if self.use_svm:
        return c
      d2 = self.d2(c)
      d3 = self.d3(d2)
      d4 = self.d4(d3)

      return self.softmax(self.acti(d4))
      # return  c
      