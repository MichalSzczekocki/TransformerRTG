import torch

def get_all_preds(model, loader, device):
	with torch.no_grad():
		all_preds = torch.tensor([])

		for batch_idx, (data, target) in enumerate(loader):
			data, target = data.to(device), target.to(device)
			preds = model(data)
			all_preds = torch.cat((all_preds, preds.data.cpu()), dim=0)

		return all_preds