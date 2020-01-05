import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import helper

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.FashionMNIST('~/.pytorch/FashionMNIST/',
                                 download=True,
                                 train=True,
                                 transform=transform)

testset = datasets.FashionMNIST(
    '~/.pytorch/F_MNIST_data/',
    download=True,
    train=False,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=True
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

images, labels = next(iter(trainloader))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        input_size = 784
        hidden_layers = [784, int(784 / 2)]
        output_size = 10

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# THis is the image that we want to show...

model = Model()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
optimizer.zero_grad()


epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

dataiter = iter(testloader)
images, labels = dataiter.next()

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(images[1]))

# Plot the image and probabilities
helper.view_classify(images[1], ps, version='Fashion')
