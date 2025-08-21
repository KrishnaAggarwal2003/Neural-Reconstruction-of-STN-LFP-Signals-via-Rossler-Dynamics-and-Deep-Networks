import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

def training_model(model, data, targets, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs+1)):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch  % 10 == 0:
            print(f'Epochs: {epoch}/{num_epochs}, Loss value: {loss.item():.5f}')

    print(f'Total number of epochs: {num_epochs}, Training completed')