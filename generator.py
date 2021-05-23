import glob
import os
import numpy as np
import random
import shutil
from PIL import Image

def main():
    weather_set = ['cloudy', 'rain', 'shine', 'sunrise']
    ratio = [0.4, 0.3, 0.3]

    raw_data_dir = 'raw_data'
    target_dir = 'data'

    for weather in weather_set:
        images_list = glob.glob(os.path.join(raw_data_dir, weather, '*.jpg'))
        random.shuffle(images_list)
        for i in range(len(images_list)):
            image = Image.open(images_list[i])
            image = np.array(image)
            if image.shape[-1] != 3:
                 continue
            category = ''
            if i < int(len(images_list) * 0.4):
                category = 'train'
            elif i < int(len(images_list) * 0.7):
                category = 'val'
            else:
                category = 'test'
            shutil.copyfile(images_list[i], os.path.join(target_dir, category, weather, os.path.basename(images_list[i])))

if __name__ == '__main__':
    main()

