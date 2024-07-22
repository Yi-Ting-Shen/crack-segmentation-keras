import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_preprocessing import load_data, split_data, DataGenerator
from config import IMG_SIZE, BATCH_SIZE, DATA_PATH

def predict(image_path, model_path):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict crack segmentation.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to be predicted.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    args = parser.parse_args()

    # 對新圖像進行預測
    prediction = predict(args.image_path, args.model_path)

    # 顯示預測結果
    plt.imshow(prediction[0, :, :, 0], cmap='gray')
    plt.title('Prediction')
    plt.show()

    # 評估一批驗證數據
    img_paths, mask_paths = load_data(DATA_PATH)
    _, val_img_paths, _, val_mask_paths = split_data(img_paths, mask_paths)
    val_gen = DataGenerator(val_img_paths, val_mask_paths, BATCH_SIZE, IMG_SIZE, aug=False, shuffle=False)
    batch_idx = np.random.randint(len(val_gen))
    imgs, masks = val_gen[batch_idx]
    preds = model.predict(imgs)

    # 顯示結果
    img_idx = np.random.randint(len(imgs))
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(imgs[img_idx])
    plt.title('Input Image')
    plt.subplot(1, 3, 2)
    plt.imshow(masks[img_idx, :, :, 0], cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(preds[img_idx, :, :, 0], cmap='gray')
    plt.title('Prediction')
    plt.show()

    # 根據不同門檻值顯示模型預測
    mask_pred_raw = preds[img_idx, :, :, 0]
    plt.imshow(masks[img_idx, :, :, 0], cmap='gray') # 先印出 GT
    plt.title('Ground Truth')
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(1, 10):
        plt.subplot(2, 5, i)
        threshold = i * 0.1  # 門檻值
        mask_threshold = mask_pred_raw.copy()
        # 下面 2 行是做二值化，讓整個 tensor 只剩 0 或 1
        mask_threshold[mask_threshold <= threshold] = 0.
        mask_threshold[mask_threshold > threshold] = 1.
        plt.imshow(mask_threshold, cmap='gray')
        plt.title(f'threshold: {threshold:.1f}')
    plt.show()
