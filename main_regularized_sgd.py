import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from low_rank_compression.proximal_operators import (
    SingularValuesEntropyApplier,
    NuclearNormApplier,
)
from low_rank_compression.models import MLP
import argparse


parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--regularizer", type=str, default="none", help="Regularizer to use"
)
args = parser.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = MLP().to("cuda")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
matrix_params = [param for name, param in model.named_parameters() if "weight" in name]


class _None:
    def step(self):
        pass


class SVEntropyAndNuclearNorm:
    def __init__(self, matrix_params):
        self.matrix_params = matrix_params
        self.reg1 = SingularValuesEntropyApplier(matrix_params, lr=0.2)
        self.reg2 = NuclearNormApplier(matrix_params, 0.01, 0.01)

    def step(self):
        self.reg1.step()
        self.reg2.step()


regularizers = {
    "none": _None(),
    "sv_entropy": SingularValuesEntropyApplier(matrix_params, lr=0.2),
    "nuclear_norm": NuclearNormApplier(matrix_params, 0.01, 0.02),
    "both": SVEntropyAndNuclearNorm(matrix_params),
}

regularizer = regularizers[args.regularizer]

epochs = 20


def entropy(x):
    smx = torch.nn.functional.softmax(x)
    return -torch.sum(smx * torch.log(smx.clamp_min(1e-12)))


for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        regularizer.step()
        if batch_idx % 100 == 0:
            # extract svd of all weight matrices
            singular_values = [torch.svd(param)[1] for param in matrix_params]
            print(
                f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
            )
            print(
                f"Average singular values: {[s.mean().item() for s in singular_values]}"
            )
            print(
                f"Entropy of singular values: {[entropy(s).item() for s in singular_values]}"
            )

            print("---------------------------------")

    # print singular values at the end of each epoch
    singular_values = [torch.svd(param)[1] for param in matrix_params]
    print("Singluar values at the end of epoch", epoch + 1)
    for i, s in enumerate(singular_values):
        print(f"Layer {i+1}: {s})")
    print("---------------------------------")


model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to("cuda"), target.to("cuda")
        output = model(data)
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), f"mnist_model_{args.regularizer}.pth")
