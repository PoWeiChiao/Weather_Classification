import glob
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import get_mobilenetV2

def predict(net, device, image_path, image_transforms):
    image = Image.open(image_path)
    image = image_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device=device)

    net.eval()
    with torch.no_grad():
        pred = net(image)
        pred = np.array(pred.data.cpu()[0])
        return pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    net = get_mobilenetV2(num_classes=4)
    net.load_state_dict(torch.load('saved/model_61.pth', map_location=device))
    net.to(device=device)

    test_dir = 'data/test'
    label_set = ['cloudy', 'rain', 'shine', 'sunrise']
    image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images_list = []
    for label in label_set:
        images_list.extend(glob.glob(os.path.join(test_dir, label, '*.jpg')))

    results = np.zeros((4, 4))
    for image_path in images_list:
        pred = predict(net, device, image_path, image_transforms)
        for i, label in enumerate(label_set):
            if label in os.path.basename(image_path):
                results[i][pred.argmax(0)] += 1
    print(results)

if __name__ == '__main__':
    main()
    


