import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 1.自定义数据集
class MyDataset(Dataset):
    def __init__(self, dirname='./train', transform=None):
        super(MyDataset, self).__init__()
        self.classes = os.listdir(dirname)
        self.images = []
        self.transform = transform
        for i, classes in enumerate(self.classes):
            classes_path = os.path.join(dirname, classes)
            for image_name in os.listdir(classes_path):
                self.images.append((os.path.join(classes_path, image_name), i))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name, classes = self.images[index]
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        return image, classes
    
train_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(size=(256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(size=(256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = MyDataset('./data/train', train_transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
val_dataset = MyDataset('./data/val', val_transform)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)

# 2. 加载预训练模型
model = models.resnet18(pretrained=True)
print(model)

# 3. 冻结预训练模型
only_train_fc = True
if only_train_fc:
    for param in model.parameters():
        param.requires_grad_(False)

# 4. 添加新层
fc_in_features = model.fc.in_features
model.fc = torch.nn.Linear(fc_in_features, 2, bias=True)

for i in model.parameters():
    if i.requires_grad:
        print(i)

# 5. 训练
epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(lr=0.001, params=model.parameters())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)
opt_step = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
max_acc = 0
epoch_acc = []
epoch_loss = []
type_ids = {0:"train", 1:"val"}
            
for epoch in range(epochs):
    for type_id, loader in enumerate([train_loader, val_loader]):
        mean_loss = []
        mean_acc = []
        for images, labels in loader:
            if type_id == 0:
                # opt_step.step()
                model.train()
            else:
                model.eval()

            images, labels = images.to(device), labels.to(device).long()

            opt.zero_grad()
            with torch.set_grad_enabled(type_id==0):
                outputs = model(images)
                _, pre_lables = torch.max(outputs, axis=1)
                loss = loss_fn(outputs, labels)

            if type_id == 0:
                loss.backward()
                opt.step()

            acc = torch.sum(pre_lables==labels) / torch.tensor(labels.shape[0], dtype=torch.float32)
            mean_acc.append(acc.cpu().detach().numpy())
            mean_loss.append(loss.cpu().detach().numpy())

        if type_id == 1:
            epoch_acc.append(np.mean(mean_acc))
            epoch_loss.append(np.mean(mean_loss))
            if max_acc < np.mean(mean_acc):
                max_acc = np.meam(mean_acc)

        print(type_ids[type_id], 'loss: ', np.mean(mean_loss), 'acc: ',  np.mean(mean_acc))

print(f'val max acc {max_acc}')

        
