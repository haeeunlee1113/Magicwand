import torch.nn as nn


class ACRNN(nn.Module):
    def __init__(self, num_classes = 4,):
        super(ACRNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # batch_size x 64 x 64 x 512
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(4,4), padding=(0,0), stride=(4,4)), # batch_size x 64 x 16 x 128
            nn.Dropout(0.4)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), #batch_size x 128 x 16 x 128
            nn.BatchNorm2d(num_features=128), 
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2,1), padding=(0,0), stride=(2,1)), # batch_size x 128 x 8 x 128
            nn.Dropout(0.4)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), #batch_size x 256 x 8 x 128
            nn.BatchNorm2d(num_features=256),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2,1), padding=(0,0), stride=(2,1)), # batch_size x 256 x 4 x 128
            nn.Dropout(0.4)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), #batch_size x 256 x 4 x 128
            nn.BatchNorm2d(num_features=256),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2,1), padding=(0,0), stride=(2,1)), # batch_size x 256 x 2 x 128
            nn.Dropout(0.4)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), #batch_size x 512 x 2 x 128
            nn.BatchNorm2d(num_features=512),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2,1), padding=(0,0), stride=(2,1)), # batch_size x 512 x 1 x 128
            nn.Dropout(0.4)
        )    

        self.LSTM1 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, num_layers=1, bidirectional=True)
        
        # attention layer
        self.a_fc1 = nn.Linear(128, 1)  
        self.a_fc2 = nn.Linear(1, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.reshape(x.size(0), -1, 128)
        x, _ = self.LSTM1(x)

        v = self.tanh(self.a_fc1(x))                  
        alphas = self.softmax(self.a_fc2(v).squeeze())          
        x = (alphas.unsqueeze(2) * x).sum(axis=1)      

        x = F.silu(self.fc1(x))
        return self.softmax(self.fc2(x))