import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from src.utils import BackdoorCrossStamp


def mnist_test():
    mnist = datasets.MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    img_mnist, _ = mnist[0]

    stamp_mnist = BackdoorCrossStamp(
        image_shape=img_mnist.shape,
        cross_size=8,
        pos=(2, 2),
        color=(1.0,),
        line_width=2,
    )

    stamped_img_mnist = stamp_mnist.stamp(img_mnist)
    save_image(stamped_img_mnist, "./imgs/stamped_mnist.png")


def batch_mnist_test():
    mnist = datasets.MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(mnist, batch_size=16, shuffle=False)
    imgs, _ = next(iter(dataloader))
    stamp_mnist = BackdoorCrossStamp(
        image_shape=imgs[0].shape,
        cross_size=8,
        pos=(2, 2),
        color=(1.0,),
        line_width=2,
    )

    stamped_imgs = stamp_mnist.stamp_batch(imgs)

    grid = make_grid(stamped_imgs, nrow=4)
    save_image(grid, "./imgs/stamped_batch_mnist.png")

def cifar10_test():
    cifar10 = datasets.CIFAR10(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    img_cifar10, _ = cifar10[0]

    stamp_cifar10 = BackdoorCrossStamp(
        image_shape=img_cifar10.shape,
        cross_size=8,
        pos=(2, 2),
        color=(1.0, 0.0, 0.0),
        line_width=2,
    )

    stamped_img_cifar10 = stamp_cifar10.stamp(img_cifar10)
    save_image(stamped_img_cifar10, "./imgs/stamped_cifar10.png")


if __name__ == "__main__":
    mnist_test()
    batch_mnist_test()
    cifar10_test()
