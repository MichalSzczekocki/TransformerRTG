import torch
from . import validate

# model.eval() is a kind of switch for some specific layers/parts
# of the model that behave differently during training and inference
# (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc.
# You need to turn off them during model evaluation, and .eval()
# will do it for you. In addition, the common practice for
# evaluating/validation is using torch.no_grad()
# in pair with model.eval() to turn off gradients computation

def calculate_model_accuracy(model, device, data_loader, title):
    model.eval()
    with torch.no_grad():
        accuracy = float(validate.validate(model, device, data_loader))
        print(title, "accuracy:", accuracy, "%")