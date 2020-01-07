import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import helper
import matplotlib.pyplot as plt

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

        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.log_softmax(self.fc3(x), dim=1))
        return x


# THis is the image that we want to show...

model = Model()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
optimizer.zero_grad()


epochs = 5
for e in range(epochs):
    train_loss = 0
    accuracy = 0
    test_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    else:
        with torch.no_grad():
            model.eval()  # put in test mode
            for images, labels in testloader:
                ps = model(images)
                loss = criterion(ps, labels)
                equals = (ps.max(dim=1).indices == labels)
                accuracy += torch.mean(equals.float())
                test_loss += loss.item()
            print('Epoch: {}'.format(e))
            print('Test loss: {}'.format(test_loss / len(testloader)
                                         ))
            print('Running Loss: {}'.format(train_loss / len(trainloader)))
            print(f'Accuracy: {accuracy.item()/len(testloader)*100}%')
            plt.plot(train_loss, label='training loss')
            plt.plot(test_loss, label='test loss')

            model.train()  # Go back to training mode

plt.show()

model.eval()
dataiter = iter(testloader)
images, labels = dataiter.next()

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(images[1]))

# Plot the image and probabilities
helper.view_classify(images[1], ps, version='Fashion')
