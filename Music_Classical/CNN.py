import os
import librosa
import numpy as np
import re
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks  # type: ignore
from tensorflow.keras import layers, models, Input
from keras.models import Model
from keras import *
from keras.src.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dropout, Dense
from PIL import Image

            
import pandas as pd
data_path = "D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\genres_original"
genres = os.listdir(data_path)

# 处理音频变成梅尔频谱图，其实还提取了色度和节拍特征，但是没有用上
X = []
y = []
M = []
C = []
# Load audio files and extract features
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        try:
            # 尝试加载音频文件
            y_data, sr = librosa.load(file_path, sr=None)  # sr=None 会使用音频的原生采样率
            # 提取梅尔频谱图
            mel_spec = librosa.feature.melspectrogram(y=y_data, sr=sr, n_mels=128,fmax=8000)
            # 转换为分贝表示
            mel = librosa.power_to_db(mel_spec)
            # 提取色度特征
            chroma_stft = librosa.feature.chroma_stft(y=y_data, sr=sr, n_fft=2048, hop_length=512)
            #提取节拍特征
            tempo, _ = librosa.beat.beat_track(y=y_data, sr=sr)

            # 调整大小为固定形状（这里使用256x256，你可以根据需要调整大小）
            chroma_stft = librosa.util.fix_length(chroma_stft, size=256, axis=1)
            #mel_spec = librosa.util.fix_length(mel_spec, size=256, axis=1)

            
            #绘制并保存频谱图
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')
            # 使用正则表达式匹配文件路径中的流派和文件名
            match = re.search(r'genres_original\\([^\\]+)\\([^\\]+)\.wav', file_path)

            if match:
                genre = match.group(1)
                file_name = match.group(2)
                number_match = re.search(r'\.(\d+)$', file_name)
                file_number = number_match.group(1)
                # 构建流派对应的子文件夹路径
                save_dir_genre = os.path.join("D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\Train_Spectogram_Images", genre)
                # 确保流派对应的子文件夹已创建
                if not os.path.exists(save_dir_genre):
                    os.makedirs(save_dir_genre)
                # 保存图像文件到流派对应的子文件夹中
                plt.savefig(
                    os.path.join(save_dir_genre, file_number + ".jpg"),
                    bbox_inches=None, pad_inches=0)
                plt.close()
            else:
                print("Error")            
            
            # 将特征添加到特征矩阵中
            X.append(mel)
            M.append(chroma_stft)
            C.append(tempo)
            # 将标签添加到标签列表中
            y.append(genre)
        except Exception as e:
            # 如果遇到错误，打印警告并跳过该文件
            print(f"Warning: Failed to load {file_path}. Error: {e}\n")
            continue






#图像切割部分
import re
import random
from PIL import Image

#下为数据预处理的切片部分

def slice_and_split_dataset(verbose=0):
    # 定义原始数据集路径和目标路径
    original_dataset_dir = "D:/smartimfor/智能信息处理/智能信息处理/Myclassification/Data/Train_Spectogram_Images"
    base_dir = "D:/smartimfor/智能信息处理/智能信息处理/Myclassification/Data/Slice_Image"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 列出原始数据集中的所有类别
    classes = os.listdir(original_dataset_dir)

    # 创建训练集和测试集的子文件夹
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # 遍历原始数据集中的每个类别
    for cls in classes:
        # 获取该类别下的所有图像文件
        images = os.listdir(os.path.join(original_dataset_dir, cls))
        # 随机打乱图像文件顺序
        random.shuffle(images)
        # 计算划分的索引位置（这里假设划分比例为 0.8）
        split_index = int(0.8 * len(images))
        # 将图像文件划分到训练集和测试集中
        train_images = images[:split_index]
        test_images = images[split_index:]

        # 开始切割和保存图像
        # 针对训练集
        for img in train_images:
            src = os.path.join(original_dataset_dir, cls, img)
            dst_folder = os.path.join(train_dir, cls)
            # 如果子文件夹不存在，创建
            os.makedirs(dst_folder, exist_ok=True)
            dst = os.path.join(dst_folder, img)
            # 执行切割和保存
            slice_and_save_image(src, dst)

        # 针对测试集
        for img in test_images:
            src = os.path.join(original_dataset_dir, cls, img)
            dst_folder = os.path.join(test_dir, cls)
            # 如果子文件夹不存在，创建
            os.makedirs(dst_folder, exist_ok=True)
            dst = os.path.join(dst_folder, img)
            # 执行切割和保存
            slice_and_save_image(src, dst)

    print("Dataset sliced and split successfully!")


def slice_and_save_image(src, dst):
    img = Image.open(src)
    subsample_size = 128
    width, height = img.size
    number_of_samples = width // subsample_size
    for i in range(number_of_samples):
        start = i * subsample_size
        img_temporary = img.crop((start, 0, start + subsample_size, subsample_size))
        img_temporary.save(dst.replace('.jpg', f'_{i}.jpg'))


#slice_and_split_dataset()



#下为加载选定文件夹内已经被分好的数据集的被切好的图像，组成输入数据
train_x = []
train_y = []
test_x = []
test_y = []
data_path = "D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\Slice_Image\\train"
files = os.listdir(data_path)
for f in files:
    genre_path = os.path.join(data_path, f)
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        try:
            img=Image.open(file_path)
            img = img.convert('L')
            img_array= np.array(img)

            #tempImg = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            train_x.append(img_array)
            train_y.append(f)

        except Exception as e:
            # 如果遇到错误，打印警告并跳过该文件
            print(f"Warning: Failed to load {file_path}. Error: {e}\n")
            continue


data_path = "D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\Slice_Image\\test"
files = os.listdir(data_path)
for f in files:
    genre_path = os.path.join(data_path, f)
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        try:
            img = Image.open(file_path)
            img = img.convert('L')
            img_array = np.array(img)
            # tempImg = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            test_x.append(img_array)
            test_y.append(f)


        except Exception as e:
            # 如果遇到错误，打印警告并跳过该文件
            print(f"Warning: Failed to load {file_path}. Error: {e}\n")
            continue
train_x = np.array(train_x)
print(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)

from keras.utils import to_categorical

label_encoder = LabelEncoder()
test_y = label_encoder.fit_transform(test_y)
train_y = label_encoder.fit_transform(train_y)
# 对训练集目标变量进行独热编码
train_y = to_categorical(train_y)

# 对测试集目标变量进行独热编码
test_y = to_categorical(test_y)
np.set_printoptions(threshold=np.inf)





print("test_y的维度为",test_y.shape)
print("train_y的维度为",train_y.shape)
print("train_x的维度为",train_x.shape)
print("test_x的维度为",test_x.shape)
# datasetSize = 0.75, this returns 3/4th of the dataset.


# Expand the dimensions of the image to have a channel dimension. (nx128x128) ==> (nx128x128x1)
train_x = np.expand_dims(np.expand_dims(train_x, axis=-1), axis=-1)
test_x = np.expand_dims(np.expand_dims(test_x, axis=-1), axis=-1)
# Normalize the matrices.
train_x = train_x / 255.
print(train_x)
test_x = test_x / 255.







'''
#这一段是使用数据生成器生成数据进行训练，在电脑一次加载不下的情况下可以使用
from keras.src.legacy.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=(0.99, 0.99),horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255, zoom_range=(0.99, 0.99))

train = train_gen.flow_from_directory("D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\Slice_Image\\train",
                                target_size=(150,150),
                                batch_size= 60, #在一次训练迭代中可以使用的图像数量，可通过这个参数调整训练速度
                                class_mode='categorical',
                                color_mode='grayscale',
                                shuffle = True,
                                seed = 199,
                                )
classes_train = train.class_indices

val = val_gen.flow_from_directory("D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\Slice_Image\\test",
                                target_size=(150,150),
                                batch_size= 10,
                                class_mode='categorical',
                                color_mode='grayscale',
                                shuffle = True,
                                seed = 199,
                               )
classes_val = val.class_indices
'''



model = Sequential()
model.add(Conv2D(filters=64, kernel_size=[7,7], kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
# Dim = (122x122x64)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (61x61x64)
model.add(Conv2D(filters=128, kernel_size=[7,7], strides=2, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (28x28x128)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (14x14x128)
model.add(Conv2D(filters=256, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (12x12x256)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (6x6x256)
model.add(Conv2D(filters=512, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (4x4x512)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (2x2x512)
model.add(BatchNormalization())
model.add(Flatten())
# Dim = (2048)
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (1024)
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (256)
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (64)
model.add(Dense(32, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (32)
model.add(Dense(10, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))

# Dim = (8)
print (model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])


history= model.fit(train_x,train_y, epochs=60, verbose=1, validation_data=(test_x,test_y))
#score = model.evaluate(test_x, test_y, verbose=1)
#print (score)
model.save("D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\model/CNN2.h5")


# 可视化部分，可删
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
# 保存图像到文件
#plt.savefig('D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\output\\training_history.png')
plt.show()

# 自动关机 (不同操作系统的关机命令)
def shutdown():
    if os.name == 'nt':  # Windows
        os.system('shutdown /s /t 1')
    elif os.name == 'posix':  # Unix/Linux/Mac
        os.system('sudo shutdown now')

# 调用关机函数（被我注释了，不需要关机就不用）
#shutdown()