import torch
import tenseal as ts
from torchvision import datasets
import torchvision.transforms as transforms
from effhe.models.baseline_square_1c2f import ConvNet 
from effhe.constants.mnist import SEED, BATCH_SIZE
from effhe.constants.paths import SAVE_PATH, BASELINE_PATH

def get_MNIST_test_loader():
    # Load the data
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    # Load one element at a time
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return test_loader

def data_to_list(x, default_dims = [28,28]):
    """
    Creates list object from tensor
    """
    d1 = default_dims[0]
    d2 = default_dims[1]

    return x.view(d1, d2).tolist()

def get_query_data():
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    query_data = test_data.__getitem__(0)[0]
    
    return query_data

def train(model):

    if model.lower() != "mnist":
        raise NotImplementedError("Only MNIST model is compatible right now")

    torch.manual_seed(SEED)

    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    def _train(model, train_loader, criterion, optimizer, n_epochs=10):
        # model in training mode
        model.train()
        for epoch in range(1, n_epochs+1):

            train_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # calculate average losses
            train_loss = train_loss / len(train_loader)

            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        # model in evaluation mode
        model.eval()
        return model

    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = _train(model, train_loader, criterion, optimizer, 10)

    # This currently saves the model at BASELINE_PATH which
    # corresponds to MNIST. Make this work with other models later
    # Change to SAVE_PATH when that happens
    torch.save(model.state_dict(), BASELINE_PATH)


