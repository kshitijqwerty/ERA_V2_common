import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Resnet Like model
        
        # prep
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # layer 1
        self.layer1_x = Model.conv_block(64,128,3,1,1)
        self.layer1_res = Model.res_block(128,128,padding=1)
        
        # layer 2
        self.layer2 = Model.conv_block(128,256,3,1,1)
        
        # layer 3
        self.layer3_x = Model.conv_block(256,512,3,1,1)
        self.layer3_res = Model.res_block(512,512,padding=1)
        
        self.max_pool = nn.MaxPool2d(kernel_size=4)

        self.fc = nn.Linear(512, 10)
        
    @staticmethod
    def conv_block(channel_in, channel_out, kernel=3, stride=1, padding=0, bias=False):
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel, stride=stride, padding=padding, bias=bias),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(channel_out),
            nn.ReLU()
        )
    
    @staticmethod
    def res_block(channel_in, channel_out, kernel=3, stride=1, padding=0, bias=False):
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.Conv2d(channel_out, channel_out, kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(channel_out),
            nn.ReLU()
        )
    
    @staticmethod
    def print_summary(model):
        summary(model.to("cpu"), input_size=(3, 32, 32))
    
           
    def forward(self, x):        
        x = self.prep(x)
        x = self.layer1_x(x)
        x = x + self.layer1_res(x)
        x = self.layer2(x)
        x = self.layer3_x(x)
        x = x + self.layer3_res(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x) # FC layer
        return F.log_softmax(x, dim=-1)
    
    