import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from src.task import MNISTNet, CifarNet

def mnist_net_test():
    mnist = datasets.MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(mnist, batch_size=16, shuffle=False)
    imgs, _ = next(iter(dataloader))
    net = MNISTNet()
    net.eval()
    with torch.no_grad():
        out = net(imgs)
    if out.shape[0] != imgs.shape[0]:
        raise ValueError(f"Shapes should match: {out.shape} == {imgs.shape}")
    if out.shape[1] != 10:
        raise ValueError(f"Out dim should be 10. Instead {out.shape[1]}")

def cifar10_net_test():
    mnist = datasets.CIFAR10(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(mnist, batch_size=16, shuffle=False)
    imgs, _ = next(iter(dataloader))
    net = CifarNet()
    net.eval()
    with torch.no_grad():
        out = net(imgs)
    if out.shape[0] != imgs.shape[0]:
        raise ValueError(f"Shapes should match: {out.shape} == {imgs.shape}")
    if out.shape[1] != 10:
        raise ValueError(f"Out dim should be 10. Instead {out.shape[1]}")

if __name__ == "__main__":
    mnist_net_test()
    cifar10_net_test()
    print("all test passed")
