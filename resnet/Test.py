import torch
import torchvision
import torch.nn as nn
import gc
import resnet

# prepare data
batch_size = 16
num_of_classes = 10 

# train_dataset = torchvision.datasets.CIFAR10(root="./data",train=True,download=True)
# test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,download=True)
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,)
test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
train_data = iter(train_data_loader)
test_data = iter(test_data_loader)
# make model
model = resnet.ResNet([2,2,3,2],resnet.IdentityBlock,num_of_classes=num_of_classes)

# make loss and optimizer
loss = nn.CrossEntropyLoss()

epochs = 20
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(),learning_rate,weight_decay=0.001,momentum=0.9)

# train loop 
for epoch in range(epochs):
    l = None
    for i, (batch,batch_labels) in enumerate(train_data):
        predictions = model(batch)
        l = loss(predictions,batch_labels)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        del predictions
        gc.collect()
    
    if epoch%10 == 0:
        # print loss
        print(f'for epoch : {epoch} the loss : {l.item()}')
        
        
with torch.no_grad():
    correct = 0 
    total = 0
    for i, (batch,batch_labels) in enumerate(test_data):
        predictions = model(batch)
        total += batch_labels.shape[0]
        _, predicted = torch.max(predictions.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    print(" accuracy : "+ str(correct/total))