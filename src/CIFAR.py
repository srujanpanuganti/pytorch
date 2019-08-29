import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        # Affine operation y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 6x6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

        # if the size is a square, you can only specify single number
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))

        x = x.view(-1, 16*5*5)
        # x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x


    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

# print(net)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def imshow(img):
    img = img /2 + 0.5  ### unnormalizing
    npimg =img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


dataiter = iter(trainloader)
images, labels = dataiter.__next__()

imshow(torchvision.utils.make_grid(images))
# plt.show()

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)


for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        # obtaining the inputs
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data

        # print(i,data)
        ## zeroing the gradients
        optimizer.zero_grad()

        ## forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        ## back prop
        loss.backward()

        ## optimization
        optimizer.step()


        ## printing stats
        running_loss += loss.item()
        if i%2000 == 1999:
            print('[%d. %5d] loss: %.3f' %(epoch +1, i +1, running_loss/2000))
            running_loss = 0.0

print('finished training')



correct = 0
total=0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

