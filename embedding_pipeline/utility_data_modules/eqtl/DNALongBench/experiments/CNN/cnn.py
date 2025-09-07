import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout_rate, use_max_pool=False):
        super().__init__()
        if use_max_pool:
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  padding=(kernel_size - 1) * dilation,
                                  stride=stride),
                nn.Dropout(p=dropout_rate),
                nn.BatchNorm1d(num_features=out_channels),
                nn.MaxPool1d(2),
                nn.ReLU(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  padding=(kernel_size - 1) * dilation,
                                  stride=stride),
                nn.Dropout(p=dropout_rate),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4, channels = [16,32,64], output_shape=2, input_length=450000, dropout_rate=0, use_max_pool=False, kernel_sizes = None, dilation_sizes = None, strides=None, task='TISP'):
        super(SimpleCNN, self).__init__()

        kernel_sizes = [3 for i in range(len(channels))] if kernel_sizes is None else kernel_sizes
        dilation_sizes = [1 for i in range(len(channels))] if dilation_sizes is None else dilation_sizes 
        strides = [1 for i in range(len(channels))] if strides is None else strides 

        layers = [ConvBlock(in_channels=in_channels, out_channels=channels[0], kernel_size=kernel_sizes[0], stride=strides[0], dilation=dilation_sizes[0], dropout_rate=dropout_rate, use_max_pool=use_max_pool)]
        self.output_shape = output_shape
        self.input_length = input_length
        
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[i+1], stride=strides[i+1], dilation=dilation_sizes[i+1], dropout_rate=dropout_rate, use_max_pool=use_max_pool))
  
        self.layers = nn.Sequential(*layers)
        self.task = task
        self.mode = 0
        if self.task=='eQTLP' or self.task == 'ETGP' or isinstance(output_shape, int):  
            self.mode = 1
            self.fc = nn.Linear(channels[-1], output_shape)
            self.relu = nn.ReLU()
        elif self.task=='RSAP' or self.task=='TISP':  
            self.mode = 2
            self.adaptive_pool = nn.AdaptiveMaxPool1d(output_shape[0])  # Adaptive pooling to ensure the exact sequence length
            self.final_conv = nn.Conv1d(channels[-1], output_shape[1], kernel_size=1)  # Adjust channels without changing length
            

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.layers(x)
        if self.mode == 1:
            x = F.max_pool1d(x, x.size(2)).squeeze()
            x = self.fc(x)
            x = self.relu(x)
        elif self.mode == 2: 
            x = self.adaptive_pool(x)
            x = self.final_conv(x)
            if self.task=='RSAP':
                x = x.transpose(1, 2) 
        return x



class Symmetrize2D(nn.Module):
    def forward(self, x):
        return (x + x.transpose(-1, -2)) / 2

class UpperTriu(nn.Module):
    def __init__(self, offset=2):
        super(UpperTriu, self).__init__()
        self.offset = offset

    def forward(self, x):
        _, _, dim1, dim2 = x.size()

        triu_indices = torch.triu_indices(dim1, dim2, offset=self.offset).to(x.device)
        x = x[:, :, triu_indices[0], triu_indices[1]]
        return x

class Crop2D(nn.Module):
    def __init__(self, crop_size):
        super(Crop2D, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, self.crop_size:-self.crop_size, self.crop_size:-self.crop_size]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution tower
        self.conv_tower = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=11, stride=4, padding=3),  
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=4, padding=3), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=4, padding=3), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=11, stride=4, padding=3), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=3),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=3),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=6),  # Output: (4, 64, 512)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Bottleneck layer
        self.bottleneck = nn.Conv1d(64, 64, kernel_size=1)

        # 2D convolution tower
        self.conv2d_tower = nn.Sequential(
            nn.Conv2d(65, 16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=11, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # # Final linear transformation
        # self.final_conv = nn.Conv2d(64, 5, kernel_size=1)

        self.crop2d = Crop2D(crop_size=32)
        self.upper_triu = UpperTriu(offset=2)
        
        # Final linear transformation
        self.final_conv =  nn.Conv1d(64, 1, kernel_size=1)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: torch.Size([4, 4, 1048576])
        x = self.conv_tower(x)  # torch.Size([4, 64, 512])
        x = self.bottleneck(x)

        # Convert to 2D
        x = x.unsqueeze(2)  # Add a dimension for the pairwise averaging  # torch.Size([4, 64, 1, 512])
        x = x.repeat(1, 1, x.size(-1), 1)  # Repeat to create the 2D map   # torch.Size([4, 64, 512, 512])

        # Concatenate positional encoding
        distance_encoding = torch.abs(torch.arange(x.size(2)).unsqueeze(1) - torch.arange(x.size(2)).unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(x.device) # distance torch.Size([1, 1, 512, 512])
        distance_encoding = distance_encoding.repeat(x.size(0), 1, 1, 1).float() # distance torch.Size([4, 1, 512, 512])
        x = torch.cat((x, distance_encoding), dim=1) # torch.Size([4, 65, 512, 512])

        x = self.conv2d_tower(x) # torch.Size([4, 64, 512, 512])

        x = self.crop2d(x) # torch.Size([4, 64, 448, 448])
        x = self.upper_triu(x) # torch.Size([4, 64, 99681])

        x = self.final_conv(x) # torch.Size([4, 1, 99681])
        return x.squeeze()


