import numpy as np
from tensorflow.keras import callbacks, optimizers
from data_preprocessing import DataGenerator, load_data, split_data
from model import build_model
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from config import IMG_SIZE, BATCH_SIZE, EPOCHS, DATA_PATH

if __name__ == "__main__":
    # 載入和預處理資料
    img_paths, mask_paths = load_data(DATA_PATH)
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = split_data(img_paths, mask_paths)

    # 建立資料生成器
    train_gen = DataGenerator(train_img_paths, train_mask_paths, BATCH_SIZE, IMG_SIZE, aug=True)
    val_gen = DataGenerator(val_img_paths, val_mask_paths, BATCH_SIZE, IMG_SIZE, aug=False, shuffle=False)

    # 建立模型
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_model(input_shape)

    # 定義自訂的 Dice coefficient 和 Dice loss
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    model.compile(optimizer=optimizers.Adam(), loss=dice_loss, metrics=[dice_coef])

    # 設定callbacks
    weight_saver = callbacks.ModelCheckpoint('seg.h5', monitor='val_loss', save_best_only=True)
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=20)

    # 訓練模型
    logs = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[weight_saver, earlystop])

    # 繪製訓練 history 紀錄
    history = logs.history
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.show()
    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    plt.title('Dice Coefficient')
    plt.show()
