import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import WeatherDataset
from model import AlexNet, get_mobilenetV2

def train(net, device, dataset_train, dataset_val, batch_size=4, epochs=100, lr=1e-3):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    writer = SummaryWriter('runs/exp_1')

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        print('running epoch {}'.format(epoch))

        net.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)

            pred = net(images)
            loss = criterion(pred, labels)
            train_loss += loss.item() * images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        for images, labels in tqdm(val_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)

            pred = net(images)
            loss = criterion(pred, labels)
            val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, val_loss))
        writer.add_scalar(tag='train loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='val loss', scalar_value=val_loss, global_step=epoch)
        torch.save(net.state_dict(), 'saved/model_{}.pth'.format(epoch))
        if val_loss <= best_loss:
            torch.save(net.state_dict(), 'saved/model_best.pth')
            best_loss = val_loss
            print('model saved')
    writer.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    net = get_mobilenetV2(num_classes=4)
    if os.path.isfile('saved/model_best.pth'):
        net.load_state_dict(torch.load('saved/model_best.pth', map_location=device))
    # net = AlexNet(in_features=3, num_classes=4)
    net.to(device=device)

    train_dir = 'data/train'
    val_dir = 'data/val'
    label_set = ['cloudy', 'rain', 'shine', 'sunrise']
    image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_train = WeatherDataset(train_dir, label_set, image_transforms)
    dataset_val = WeatherDataset(val_dir, label_set, image_transforms)
    train(net, device, dataset_train, dataset_val)

if __name__ == '__main__':
    main()