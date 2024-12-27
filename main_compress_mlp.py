from low_rank_compression.models import MLP
from low_rank_compression.modules import to_low_rank
import argparse

# mnist
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)
mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

argparse = argparse.ArgumentParser(description="PyTorch MNIST Example")
argparse.add_argument(
    "--regularizer", type=str, default="none", help="Regularizer to use"
)

args = argparse.parse_args()
# torch.save(model.state_dict(), f'mnist_model_{args.regularizer}.pth')
model_path = f"mnist_model_{args.regularizer}.pth"

# eval previous accuracy
model = MLP().to("cuda")
model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in mnist_test_loader:
        images, labels = data
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")


ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for ratio in ratios:
    model = MLP().to("cuda")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model = to_low_rank(model, ratio=ratio)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in mnist_test_loader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total}% with ratio {ratio}"
    )
