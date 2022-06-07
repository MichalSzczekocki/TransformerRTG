import torch
import matplotlib.pyplot as plt
from torch import utils

def show_image(device, dataset):
    test_loader = utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        print(target)

        plt.imshow(data[0][0].cpu(), cmap='gray')
        plt.show()