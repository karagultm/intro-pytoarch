#1 imports
import torch
import torch.nn as nn 
import torch.nn.functional as F #nonlinearilization ReLU
import torch.optim as optim #Adam, adagard etc

from torch.utils.data import  DataLoader #minibatches, shuffling, sgd etc

import torchvision.datasets as datasets #dataset mnist
import torchvision.transforms as transforms #convertion to tensor

#2 creat FCNN
class NN (nn.Module):
    def __init__(self, input_size,num_classes): #28x28=784 input size, 10 classes
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(1,1),padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) #flattening the data 
        x = self.fc1(x)
        return x

#3 set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#4 set the hyperparameters
input_size= 784 #28x28
num_classes = 10 #0-9
learning_rate = 0.01
batch_size= 64
num_epochs=1

#5 load the data
train_dataset = datasets.MNIST(root= './data', train=True, transform=transforms.ToTensor(),download=True) #root datayı yükleyeceğimiz yer
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #batch size 64, shuffle=True 
# bu üstteki kısım datadan batch kadar alır ve onları datadan siler. shuffle ile de random olarak alır bunları
#batch size x channel x row x column
#64 x 1 x 28 x 28
test_dataset = datasets.MNIST(root= './data', train=False, transform=transforms.ToTensor(),download=True) #root datayı yükleyeceğimiz yer
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#6 initalize

model = NN(input_size=input_size, num_classes=num_classes)

#7 loss function and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(),lr=learning_rate) #adam optimizer

#8 training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #sending data and targets to device (gpu)
        data= data.to(device= device)
        targets= targets.to(device= device)
        
        # data = data.reshape(data.shape[0], -1) #flattening the data no need for this time
        scores = model(data) #forward pass
        loss = criterion(scores, targets)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()


#test kısmını salıyor..