import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from tensorflow.keras import utils
from config import IMG_SIZE, BATCH_SIZE, DATA_PATH

class DataGenerator(utils.Sequence):
    def __init__(self, img_paths, mask_paths, batch_size, img_size, shuffle=True, aug=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        self.aug = aug
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                rotate=(-45, 45),
                shear=(-16, 16),
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                mode='edge',
            ),
        ])
        self.indexes = np.arange(len(self.mask_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.mask_paths) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in idxs]
        batch_mask_paths = [self.mask_paths[i] for i in idxs]
        X, y = self.__data_generation(batch_img_paths, batch_mask_paths)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths, mask_paths):
        x = np.empty((len(img_paths), self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.empty((len(img_paths), self.img_size, self.img_size), dtype=np.float32)
        for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            img = self.preprocess(img, normalize=True)
            mask = self.preprocess(mask, normalize=False)
            mask[mask > 0] = 1
            x[i] = img
            y[i] = mask
        y = np.expand_dims(y, axis=-1)
        if self.aug:
            x, y = self.seq(images=x, heatmaps=y)
        return x, y

    def preprocess(self, img, normalize):
        data = cv2.resize(img, (self.img_size, self.img_size))
        if normalize:
            data = data / 255.
        return data

def load_data(data_path):
    img_paths = sorted(glob(os.path.join(data_path, 'image/*.jpg')))
    mask_paths = sorted(glob(os.path.join(data_path, 'mask/*.png')))
    return img_paths, mask_paths

def split_data(img_paths, mask_paths, test_size=0.2):
    return train_test_split(img_paths, mask_paths, test_size=test_size)

if __name__ == "__main__":
    # 載入和分割資料
    img_paths, mask_paths = load_data(DATA_PATH)
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = split_data(img_paths, mask_paths)
    
    # 創建資料生成器
    train_gen = DataGenerator(train_img_paths, train_mask_paths, BATCH_SIZE, IMG_SIZE, aug=True)
    val_gen = DataGenerator(val_img_paths, val_mask_paths, BATCH_SIZE, IMG_SIZE, aug=False, shuffle=False)
    
    # 檢查生成的批次資料
    batch_x, batch_y = train_gen[0]
    print("批次資料 X 的shape:", batch_x.shape)
    print("批次資料 Y 的shape:", batch_y.shape)
