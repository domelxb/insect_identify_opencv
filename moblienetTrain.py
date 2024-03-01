import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 数据增强
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # 在应用数据增强前获取类名
    class_names = train_ds.class_names

    # 应用数据增强
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # 对验证集数据进行缓存和预取，提高加载效率
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds, class_names


# 构建mobilenet模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),  # 添加 Dropout 层
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model






# 展示训练过程的曲线，使用双Y轴和空心点
def show_loss_acc(history, results_dir='results', filename='result-mobilenet.txt'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig, ax1 = plt.subplots(figsize=(8, 8))

    ax2 = ax1.twinx()
    ax1.plot(epochs_range, acc, 'bo-', label='Training Accuracy', fillstyle='none')
    ax1.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy', fillstyle='none')
    ax2.plot(epochs_range, loss, 'bs-', label='Training Loss', fillstyle='none')
    ax2.plot(epochs_range, val_loss, 'rs-', label='Validation Loss', fillstyle='none')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax2.set_ylabel('Loss', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Training and Validation Accuracy and Loss')
    plt.savefig(f'{results_dir}/results_mobilenet.png', dpi=100)
    plt.show()

    # 保存训练数据到文本文件
    with open(f'{results_dir}/{filename}', 'w') as f:
        f.write('Epoch,Training Accuracy,Validation Accuracy,Training Loss,Validation Loss\n')
        for i in epochs_range:
            f.write(f"{i},{acc[i]},{val_acc[i]},{loss[i]},{val_loss[i]}\n")



# train函数，训练模型
def train(epochs):


    begin_time = time()
    train_ds, val_ds, class_names = data_load("../split_data/train", "../split_data/val", 224, 224, 16)
    num_classes = len(class_names)
    model = model_load(IMG_SHAPE=(224, 224, 3), class_num=num_classes)
    # visualize_model_parameters_custom(model, 'mymodelperformance.png')



    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])
    model.save("models/mobilenet_fv_2.h5")

    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")

    show_loss_acc(history)

if __name__ == '__main__':
    train(epochs=30)