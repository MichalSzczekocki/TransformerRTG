import torch

def validate(model, device, data_loader):
    total = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        predicted = torch.max(output.data, 1)[1]
        total += len(target)
        correct += torch.sum(predicted == target)
    return 100 * correct / float(total)