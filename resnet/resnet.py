import torch
import torch.nn as nn

class IdentityBlock(nn.Module):
    # just add the input as is at the end of forward pass
    def __init__(self, input_channels,output_channels,stride=1,downsample=None):
        super(IdentityBlock,self).__init__()
        padding =1
        self.block = torch.nn.Sequential(
            nn.Conv2d(input_channels,out_channels=output_channels,kernel_size=3,stride=stride,padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=padding),
            nn.BatchNorm2d(output_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
    
    def forward(self,x):
        residue = x
        out = self.block(x)
        if self.downsample :
            residue = self.downsample(x)
        # print("out shape :"+ str(out.shape))
        # print("residue shape :"+str(residue.shape))
        out += residue
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
    # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    def __init__(self,layers,block,num_of_classes):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3,self.inplanes,7,2,3),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU()
        )
        self.initial_pool = nn.MaxPool2d(3,2,1)
        self.resnet_layer1 = self.layer_with_blocks(block,layers[0],64,stride=1)
        self.resnet_layer2 = self.layer_with_blocks(block,layers[1],128,stride=2)
        self.resnet_layer3 = self.layer_with_blocks(block,layers[2],256,stride=2)
        self.resnet_layer4 = self.layer_with_blocks(block,layers[3],512,stride=2)
        self.avg_pool = nn.AvgPool2d(7,stride=1)
        self.final = nn.Linear(512,num_of_classes)
        
    def layer_with_blocks(self,block,number_of_blocks,channels,stride=1):
        downsample = None
        if self.inplanes != channels or  stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(channels)
            )
        layers = []
        layers.append(block(self.inplanes,channels,stride,downsample))
        self.inplanes = channels
        for i in range(1,number_of_blocks):
            layers.append(block(self.inplanes,channels))
        
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.initial_conv(x)
        out = self.initial_pool(out)
        out = self.resnet_layer1(out)
        # print("out.shape after layer 1:"+str(out.shape))
        out = self.resnet_layer2(out)
        # print("out.shape after layer 2:"+str(out.shape))
        out = self.resnet_layer3(out)
        # print("out.shape after layer 3:"+str(out.shape))
        out = self.resnet_layer4(out)
        # print("out.shape after layer 4:"+str(out.shape))
        out = self.avg_pool(out)
        # print("out.shape avg pool :"+str(out.shape))
        out = out.view(out.size(0),-1)
        out = self.final(out)
        return out
        