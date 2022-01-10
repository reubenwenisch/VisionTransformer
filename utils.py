import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def download_data():
    #Load data
    os.makedirs('data', exist_ok=True)
    train_dir = 'data/train'
    test_dir = 'data/test'

    #Unzipping dataset
    with zipfile.ZipFile('train.zip') as train_zip:
        train_zip.extractall('data')
        
    with zipfile.ZipFile('test.zip') as test_zip:
        test_zip.extractall('data')
    #Creating train and test list 

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    #printing length of the dataset

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")
    return train_list, test_list

def img_show(train_list, labels):
    # printing few images 
    random_idx = np.random.randint(1, len(train_list), size=9)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for idx, ax in enumerate(axes.ravel()):
        img = Image.open(train_list[idx])
        ax.set_title(labels[idx])
        ax.imshow(img)

#Loading dataset for training 

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0
        return img_transformed, label

def train(train_loader, valid_loader, device, model, criterion, optimizer, epoch):
    #start training
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output, label)
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

# loop over the dataset multiple times
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Loss: {}'.format(running_loss)

print('Finished Training')