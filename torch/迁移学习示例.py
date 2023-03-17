import os
import time
import torch
from torchvision import models,transforms
from torch import nn,optim
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# define image transforms to do data augumentation
data_transform = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    'val':transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

# define data folder using ImageFolder to get images and classes from folder
root = './kaggle_dog_vs_cat/'
data_folder = {
    'train':
    ImageFolder(os.path.join(root,'data/train'),transform=data_transform['train']),
    'val':
    ImageFolder(os.path.join(root,'data/val'),transform=data_transform['val'])
}

# define dataloader to load images
batch_size = 32
dataloader = {
    'train':DataLoader(data_folder['train'],batch_size=batch_size,shuffle=True,num_workers=2),
    'val':DataLoader(data_folder['val'],batch_size=batch_size,num_workers=2)
}

# get train data size and validation data size
data_size = {x:len(dataloader[x].dataset for x in ['train','val'])}
# get numbers of classes
img_classes = len(dataloader['train'].dataset.classes)


# test if using GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 如果固定卷积基则为方法二，不固定卷积基则为方法一
fix_param = True  

# 加载预训练模型
transfer_model = models.resnet18(pretrained=True)
if fix_param:
    for param in transfer_model.parameters():
        param.requires_grad = False
dim_in = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(dim_in,2)
if use_gpu:
    transfer_model = transfer_model.cuda()
    
# define optimize function and loss function
if fix_param: # 只更新添加的全连接层
    optimizer = optim.Adam(transfer_model.fc.parameters(),lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
else:
    optimizer = optim.Adam(transfer_model.parameters(),lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)   

criterion = nn.CrossEntropyLoss()



def train_model(model, criterion, optimizer, scheduler=None, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.double() / data_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return best_model


best_model = train_model(model, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=10)
#best_model(test)
