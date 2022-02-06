import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

model = nn.Sequential(
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 10)
)

# More flexible model
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)
    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits

# Gordon's Note: I'm using this code on two machines - one with a GPU, one without
# So I'm adding this special code not in the tutorial
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet().to(device)

params = model.parameters()
optimizer = optim.SGD(params, lr=1e-2)

loss = nn.CrossEntropyLoss()

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train()
    for batch in train_loader:
        x, y = batch

        b = x.size(0)
        x = x.view(b, -1).to(device)

        l = model(x)  # predict

        # Gordon note: this is a better method than what's in the tutorial for sending y to device
        y = y.to(device)
        
        J = loss(l, y)
        model.zero_grad()
        J.backward()
        optimizer.step()  # actually shift the params by gradients
        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}, train loss: {torch.tensor(losses).mean():.2f}')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')

    # NOW EVALUATE

    losses = list()
    accuracies = list()
    model.eval()
    for batch in val_loader:
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1).to(device)

        with torch.no_grad():
            l = model(x)

        y = y.to(device)
        J = loss(l, y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
    print(f'Epoch {epoch + 1}', end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
