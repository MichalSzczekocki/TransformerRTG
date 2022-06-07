import torch
import torchvision
import matplotlib.pyplot as plt
import time

from torch import nn, optim, utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from config import TRAINING_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE
from utils.customized_dataset import CustomImageDataset
from plot import plot_confusion_matrix, test_and_show_errors, show_image
from utils import get_all_preds
from calculate_accuracy import calculate_accuracy, validate
from model import custom_model

# Print
test = 0
loss_list = []
iteration_list = []
accuracy_test_list = []
accuracy_validation_list = []
count = 0

stop = False
stop_value = 0.000000000000005
PATH = 'transformer_rtg.pt'

# Parameters

learning_rate = 0.000005

num_epochs = 2

momentum = 0.5

train_transform = transforms.Compose([
    transforms.Resize(size=(525, 525)),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.CenterCrop(size=500),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = transforms.Compose([
    transforms.Resize(size=(500, 500)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_train = CustomImageDataset(TRAINING_DIR, transform=train_transform)
validation_dataset = CustomImageDataset(VALIDATION_DIR, transform=transform)
dataset_test = CustomImageDataset(TEST_DIR, transform=transform)

train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# Use graphic card if available
cuda = torch.cuda.is_available()

print("Is cuda available ?", cuda)

dev = "cuda" if cuda else "cpu"

# Choose device cuda or cpu
device = torch.device(dev)

# Show each image
# show_image.show_image(device, dataset_validation)

# Create model
model = custom_model.CNNModel().to(device)

# # Load model
# # model.load_state_dict(torch.load(PATH))

# Draw model summary
summary(model, input_size=(3, 500, 500), device=dev)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Setup loss function
error = nn.CrossEntropyLoss()

# Setup scheduler (provides several methods to adjust the learning rate based on the number of epochs)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

start_time = time.time()

for epoch in range(num_epochs):
    # Tells model that is going to be trained
    model.train()

    if stop:
        break

    # In this case data = images and target = labels
    for batch_idx, (data, target) in enumerate(train_loader):

        # Transfer to GPU or CPU allows to generate your data on multiple cores in real time
        data, target = data.to(device), target.to(device)

        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation
        optimizer.zero_grad()

        # Feed network
        output = model(data)

        # Calculate loss
        loss = error(output, target)

        # Calculate change for each od weights and biases in model
        loss.backward()

        # Update weight and biases for example, the SGD optimizer performs: x += -lr * x.grad
        optimizer.step()

        count += 1

        if (batch_idx + 1) % BATCH_SIZE == 0:
            # Switch to eval mode
            model.eval()

            with torch.no_grad():
                end_time = time.time()

                accuracy_test = float(validate.validate(
                    model, device, test_loader))

                accuracy_validation = float(validate.validate(
                    model, device, validation_loader))

                print("It took {:.2f} seconds to execute this".format(
                    end_time - start_time))

                # Store loss, iteration and accuracy
                loss_list.append(loss.data.cpu())
                iteration_list.append(count)
                accuracy_test_list.append(accuracy_test)
                accuracy_validation_list.append(accuracy_validation)

                print("Epoch:", epoch + 1, "Batch:", batch_idx + 1, "Loss:",
                      float(loss.data), "Accuracy test:", accuracy_test, "%", "Accuracy validation:",
                      accuracy_validation, "%")

                if accuracy_test >= 90:
                    print("Stop condition achieved loss.data", stop_value)
                    stop = True

            # Switch to train mode
            model.train()

        if stop:
            break

    # Adjust learning rate
    scheduler.step()

# SAVE MODEL
torch.save(model.state_dict(), PATH)

# VISUALIZATION LOSS AND ACCURACY

plt.plot(iteration_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

plt.plot(iteration_list, accuracy_test_list, color="red")
plt.plot(iteration_list, accuracy_validation_list, color="blue")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()

# CALCULATE MODEL ACCURACY ON TRAIN AND TEST DATA

calculate_accuracy.calculate_model_accuracy(
    model, device, test_loader, "Final test data")
calculate_accuracy.calculate_model_accuracy(
    model, device, train_loader, "Final train data")

# DRAW CONFUSION MATRIX

train_preds = get_all_preds.get_all_preds(model, train_loader, device)
cm = confusion_matrix(dataset_train.targets, train_preds.argmax(dim=1))
plot_confusion_matrix.plot_confusion_matrix(cm, dataset_train.classes,
                                            title="Confusion matrix train data")
plt.show()

test_preds = get_all_preds.get_all_preds(model, test_loader, device)
cm = confusion_matrix(dataset_test.targets, test_preds.argmax(dim=1))
plot_confusion_matrix.plot_confusion_matrix(cm, dataset_test.classes,
                                            title="Confusion matrix test data")
plt.show()

# SHOW WRONGLY CLASSIFIED PICTURES

test_and_show_errors.test_and_show_errors(model, device, dataset_test)
