# 导入包和版本查询
import numpy as np
import torch
import torch.nn as nn
import torchvision
import PIL

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

# 可复现性
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 一 基本配置
# 显卡设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

torch.cuda.empty_cache()


# 二 张量(Tensor)处理

# 张量处理
tensor = torch.tensor(3,4,5)
tensor.type()
tensor.size()
tensor.dim()

# 命名张量
# 在PyTorch 1.3之前，需要使用注释
# Tensor[N, C, H, W]
images = torch.randn(32, 3, 56, 56)
images.sum(dim=1)
images.select(dim=1, index=0)

# PyTorch 1.3之后
NCHW = ['N', 'C', 'H','W']
images = torch.randn(32, 3, 56, 56, names=NCHW)
images.sum('C')
images.select('C', index=0)
# 也可以这么设置
tensor = torch.rand(3,4,1,2,names=('C', 'N', 'H', 'W'))
# 使用align_to可以对维度方便地排序
tensor = tensor.align_to('N', 'C', 'H', 'W')

# 数据类型转换
# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor
torch.set_default_tensor_type(torch.FloatTensor)

# 类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()

# torch.Tensor与np.ndarray转换
ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float() # If ndarray has negative stride.

# Torch.tensor与PIL.Image转换

# pytorch中的张量默认采用[N, C, H, W]的顺序，并且数据范围在[0,1]，需要进行转置和规范化
# torch.Tensor -> PIL.Image
image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0, max=255).byte().permute(1,2,0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way

# PIL.Image -> torch.Tensor
path = r'./figure.jpg'
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) # Equi

# np.ndarray与PIL.Image的转换
image = PIL.Image.fromarray(ndarray.astype(np.uint8))
ndarray = np.asarray(PIL.Image.open(path))

# 从只包含一个元素的张量中提取值
value = torch.rand(1).item()

# 张量形变
# 在将卷积层输入全连接层的情况下通常需要对张量做形变处理，
# 相比torch.view，torch.reshape可以自动处理输入张量不连续的情况。
tensor = torch.rand(2,3,4)
shape = (6, 4)
tensor = torch.reshape(tensor, shape)

# 打乱顺序
tensor = tensor[torch.randperm(tensor.size(0))]  # 打乱第一个维度

# 水平翻转
# pytorch不支持tensor[::-1]这样的负步长操作，水平翻转可以通过张量索引实现
# 假设张量的维度为[N, D, H, W].
tensor = tensor[:,:,:,torch.arange(tensor.size(3) - 1, -1, -1).long()]

# 复制张量
'''
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No                
'''

# 张量拼接

'''
注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，
而torch.stack会新增一维。例如当参数是3个10x5的张量，torch.cat的结果是30x5的张量，
而torch.stack的结果是3x10x5的张量。

tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)
'''

# 将整数标签转为one-hot编码

# pytorch的标记默认从0开始
tensor = torch.tensor([0, 2, 1, 3])
N = tensor.size(0)
num_classes = 4
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes))

from torchvision.transforms import ToTensor, Lambda
from torchvision import datasets

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
# target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


# 得到非零元素
torch.nonzero(tensor)               # index of non-zero elements
torch.nonzero(tensor==0)            # index of zero elements
torch.nonzero(tensor).size(0)       # number of non-zero elements
torch.nonzero(tensor == 0).size(0)  # number of zero elements

# 判断两个张量相等
tensor1 = ''
tensor2 = ''
torch.allclose(tensor1, tensor2)  # float tensor
torch.equal(tensor1, tensor2)     # int tensor

# 张量扩展
# Expand tensor of shape 64*512 to shape 64*512*7*7.
tensor = torch.rand(64,512)
torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)

# 矩阵乘法
# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).
result = torch.mm(tensor1, tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)
result = torch.bmm(tensor1, tensor2)

# Element-wise multiplication.
result = tensor1 * tensor2

# 计算两组数据之间的两两欧式距离
X1 = []
X2 = []
dist = torch.sqrt(torch.sum((X1[:,None,:] - X2) ** 2, dim=2))

# 三 模型定义和操作

# 一个简单两层卷积网络的示例
class ConvNet(nn.Module):
    def __init__(self, num_class=10) -> None:
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=5, stride=1, padding=2),\
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# 计算模型整体参数量
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())

# 查看网络中的参数
# 通过model.state_dict()或者model.named_parameters()函数查看现在的全部可训练参数
    
# model.modules() 和 model.children() 的区别
# model.modules() 会迭代地遍历模型的所有子层，而 model.children() 只会遍历模型下的一层
# Common practise for initialization.
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)


# 提取模型中的某一层
# 提取模型中前两层
new_model = nn.Sequential(*list(model.children())[:2])
# 提取出模型中的所有卷积层
conv_model = nn.Sequential()
for layer in model.named_modules():
    if isinstance(layer[1], nn.Conv2d):
        conv_model.add_module(layer[0], layer[1])

# 将在 GPU 保存的模型加载到 CPU
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# 导入另一个模型的相同部分到新的模型
model_new_dict = model_new.state_dict()
model_common_dict = {k:v for k, v in model_saved.items() if k in model_new_dict.keys()}
model_new_dict.update(model_common_dict)
model_new.load_state_dict(model_new_dict)

# 常用训练和验证数据图片预处理

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224,
                                             scale=(0.08, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
 ])

val_transform = torchvision.transforms.Compose([
torchvision.transforms.Resize(256),
torchvision.transforms.CenterCrop(224),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)),
])


# ===========================模型训练=========================
from torch.utils.data import DataLoader, Dataset
loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)
train_loader = DataLoader(Dataset())
epochs = 20
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # forward pass
        predictions = model(x)
        loss = loss_fn(predictions, y)

        # Backward and optimizer
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (1+1) % 100 == 0:
            print('epoch:[{}/{}], Step:[{}/{}], loss:{}'.format(epoch+1, epochs, i+1, total_step, loss.item()))

# ===========================模型验证=========================
test_loader = DataLoader(Dataset())
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Test accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# 梯度裁剪（gradient clipping）

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)

# 得到当前学习率

# If there is one global learning rate (which is the common case)
lr = next(iter(optimizer.param_groups))['lr']

# If there are multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])

# =================================学习率衰减============================

# Reduce learning rate when validation accuarcy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
for t in range(20):
    train(...)
    val(...)
    scheduler.step(val_acc)

# Cosine annealing learning rate.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

# Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
for t in range(0, 80):
    scheduler.step()    
    train(...)
    val(...)

# Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: t / 10)
for t in range(0, 10):
    scheduler.step()
    train(...)
    val(...)

# 优化器链式更新

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR
model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(4):
    print(epoch, scheduler2.get_last_lr()[0])
    optimizer.step()
    scheduler1.step()
    scheduler2.step()


# =====================模型训练可视化================
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# ==================保存与加载断点=========================
import shutil

num_epochs=20
start_epoch = 0
resume = 0
# load checkpoint
if resume:# resume为参数，第一次训练时设为0，中断再训练时设为1
    model_path = os.path.join('model', 'best_checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimier'])
    print('Load checkpoint at epoch {}.'.format(start_epoch))
    print('Best accuracy so far {}.'.format(best_acc))

best_acc = 0
for epoch in range(start_epoch, num_epochs):
    # train step
    # test step -> current_acc


    # save checkpoint
    is_best = current_acc > best_acc
    best_acc = max(current_acc, best_acc)
    checkpoint = {
        'best_acc':best_acc,
        'epoch':epoch+1,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict()
    }

    model_path = os.path.join('model', 'checkpoint.pth.tar')
    best_model_path = os.path.join('model', 'best_checkpoint.pth.tar')
    torch.save(checkpoint, model_path)
    if is_best:
        shutil.copy(model_path, best_model_path)


# 提取 ImageNet 预训练模型某层的卷积特征
import collections

# VGG-16 relu5-3 feature.
model = torchvision.models.vgg16(pretrained=True).features[:-1]
# VGG-16 fc7 feature.
model = torchvision.models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
# ResNet GAP feature.
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(collections.OrderedDict(
    list(model.named_children())[:-1]))

with torch.no_grad():
    model.eval()
    conv_representation = model(image)


# 提取 ImageNet 预训练模型多层的卷积特征

class FeatureExtractor(torch.nn.Module):
    """Helper class to extract several convolution features from the given
    pre-trained model.

    Attributes:
        _model, torch.nn.Module.
        _layers_to_extract, list<str> or set<str>

    Example:
        >>> model = torchvision.models.resnet152(pretrained=True)
        >>> model = torch.nn.Sequential(collections.OrderedDict(
                list(model.named_children())[:-1]))
        >>> conv_representation = FeatureExtractor(
                pretrained_model=model,
                layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)
    """
    def __init__(self, pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)

    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
            return conv_representation


# 微调全连接层
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 100) # Replace the last fc layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

# 以较大学习率微调全连接层，较小学习率微调卷积层
model = torchvision.models.resnet18(pretrained=True)
finetuned_parameters = list(map(id, model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'params':conv_parameters, 'lr':1e-3},\
    {'parmas':model.fc.parameters()}]

optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)

