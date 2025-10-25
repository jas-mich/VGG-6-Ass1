import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import wandb


from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import torch


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        self.p1 = p1
        self.op1=operation1
        self.magnitude_idx1=magnitude_idx1
        self.p2 = p2
        self.op2=operation2
        self.magnitude_idx2=magnitude_idx2
        self.fillcolor=fillcolor
        self.init = 0

    def gen(self, operation1, magnitude_idx1, operation2, magnitude_idx2, fillcolor):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if self.init == 0:
            self.gen(self.op1, self.magnitude_idx1, self.op2, self.magnitude_idx2, self.fillcolor)
            self.init = 1
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


def GetCifar10(batchsize):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def make_layers(cfg, activation_name="relu", batch_norm=False):
    layers = []
    in_channels = 3

    # Activation selection
    if activation_name == "relu":
        activation = nn.ReLU(inplace=True)
    elif activation_name == "tanh":
        activation = nn.Tanh()
    elif activation_name == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_name == "leaky_relu":
        activation = nn.LeakyReLU(inplace=True)
    elif activation_name == "silu":
        activation = nn.SiLU()
    elif activation_name == "gelu":
        activation = nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {activation_name}")

    # Layer construction
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation]
            else:
                layers += [conv2d, activation]
            in_channels = v

    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, activation_name="relu", batch_norm=True):
    return VGG(make_layers(cfg, activation_name=activation_name, batch_norm=batch_norm), num_classes=num_classes)





from tqdm import tqdm
def eval(model,data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        runnin_test_loss = 0.0
        for data, target in tqdm(data,desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            test_loss = criterion(outputs,target)
            runnin_test_loss +=  test_loss
    acc = 100. * correct / total
    return acc, runnin_test_loss


def train_model(model,epochs,optimizer,train_loader,test_loader,  test_acc):
    best_test_acc = test_acc

    model.train()
    for epoch in range(epochs):
      running_loss = 0.0
      for data, target in tqdm(train_loader,desc="Training single epoch"):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          outputs = model(data)
          loss = criterion(outputs, target)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print("train")
      train_acc,_ = eval(model,train_loader)
      print("test")
      test_acc, run_test_loss = eval(model,test_loader)
      avg_loss = running_loss / len(train_loader)
      avg_test_loss = run_test_loss / len(test_loader)
      wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": test_acc,
            "test_loss": avg_test_loss
        })
      print(f"Epoch {epoch} - Train_Loss: {running_loss/len(train_loader):.4f} , Train_acc: {train_acc}, Test_acc : {test_acc}")
      if test_acc > best_test_acc:
            test_acc_str = str(test_acc).replace(".","_")
            save_path=f"./outputs/bestmodel_{str(lr)}_{test_acc_str}_epoch{str(epoch)}.pth"
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            wandb.run.summary["best_test_acc"] = best_test_acc
            print(f"✅ Best model saved at epoch {epoch+1} with Test_acc: {test_acc:.2f}")
    return save_path, test_acc

if __name__=="__main__":
    import warnings
    warnings.filterwarnings(action="ignore")
    wandb.init(project="optimizer-sweep-demo2")
    # wandb.login()
    config = wandb.config
    wandb.run.name =f"{config.optimizer}_{config.activation}_lr{config.lr}_bs{config.batch_size}_ep{config.epochs}"
    
    test_Acc_Param = 0.0 #initially run
    # VGG-6 configuration
    cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
    model = vgg(cfg_vgg6, num_classes=10, activation_name=wandb.config.activation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load("outputs/bestmodel_0.001_80_0_epoch144.pth"))
    print("80% accuracy model loaded")
    model.to(device)
    #excluding to include sweep
    # model = vgg(cfg_vgg6, num_classes=10, batch_norm=True)
    print(model)

    lr = config.lr
    bz = config.batch_size #power of 2 is good for GPU
    criterion = nn.CrossEntropyLoss()
    epochs = config.epochs
    train_loader,test_loader = GetCifar10(bz)
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config.lr)
    elif config.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    elif config.optimizer == "nesterov_sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9,nesterov=True)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=config.lr)
    elif config.optimizer == "nadam":
        optimizer = optim.NAdam(model.parameters(), lr=config.lr)
    
    
    print(f"device: {device}")
    best_acc , test_acc = train_model(model,epochs,optimizer,train_loader,test_loader,test_Acc_Param)
 
