import torch
import matplotlib.pyplot as plt
from torch import utils

def test_and_show_errors(model, device, dataset_test):
    test_loader = utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False)

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            predicted = torch.max(output.data, 1)[1]

            if (predicted != target)[0]:
                print(predicted, target)
                plt.imshow(data[0][0].cpu(), cmap='gray')
                plt.show()