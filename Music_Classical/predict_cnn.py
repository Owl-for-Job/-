from keras.models import load_model
import librosa
import numpy as np
import re
import os
from PIL import Image
import matplotlib.pyplot as plt
# 加载模型
model = load_model("D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\model\CNN.h5")

import numpy as np

# 定义类别
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# 创建类别到整数的映射字典
genre_to_int = {genre: i for i, genre in enumerate(genres)}

# 创建整数到类别的反向映射字典
int_to_genre = {i: genre for genre, i in genre_to_int.items()}

# 示例：将类别转换为整数
def genre_to_integer(genre):
    return genre_to_int.get(genre, None)

# 示例：将整数转换为类别
def integer_to_genre(i):
    return int_to_genre.get(i, None)
def preprocess_wav(file_path):
    # 加载 WAV 文件
    y_data, sr = librosa.load(file_path, sr=None)
    # 提取 Mel 频谱图
    mel_spec = librosa.feature.melspectrogram(y=y_data, sr=sr, n_mels=128, fmax=8000)
    # 转换为分贝表示
    mel = librosa.power_to_db(mel_spec)
    # 绘制并保存频谱图
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = float(mel.shape[1]) / float(100)
    fig_size[1] = float(mel.shape[0]) / float(100)
    plt.rcParams["figure.figsize"] = fig_size
    plt.axis('off')
    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel, cmap='gray_r')

    # 保存图像文件到流派对应的子文件夹中
    plt.savefig(
        #提取的频谱图保存的路径
        os.path.join("D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\predict/" + "mel.jpg"),
        bbox_inches=None, pad_inches=0)
    plt.close()


    return os.path.join("D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\predict/" + "mel.jpg")

def slice_and_save_image(src):
    img = Image.open(src)
    subsample_size = 128
    width, height = img.size
    number_of_samples = width // subsample_size
    for i in range(number_of_samples):
        start = i * subsample_size
        img_temporary = img.crop((start, 0, start + subsample_size, subsample_size))
        #切片后的频谱图保存的路径
        img_temporary.save(f'D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\predict\slice/_{i}.jpg')



def predict_genre(mel_array, model,directory_path="D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\predict\Slice"):
    # 获取目录下的所有文件列表

    files = os.listdir(directory_path)

    # 初始化一个空列表，用于存储每个切片的预测结果
    predictions = []
    # 对 Mel 频谱图进行分类预测
    for filename in files:
        # 构建文件的完整路径
        file_path = os.path.join(directory_path, filename)

        try:
            # 打开图像文件
            img = Image.open(file_path)
            # 转换为灰度图像
            img = img.convert('L')
            # 将图像转换为 numpy 数组
            img_array = np.array(img)

            img_array = np.expand_dims(img_array, axis=-1)  # 添加通道维度
            img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
            img_array = img_array.astype('float32') / 255.0  # 归一化

            prediction = model.predict(img_array)

            #print(prediction)
            # 获取预测类别的索引
            predicted_class_index = np.argmax(prediction)
            # 将预测结果添加到列表中
            predictions.append(predicted_class_index)

        except Exception as e:
            # 如果遇到错误，打印警告并跳过该文件
            print(f"Warning: Failed to load {file_path}. Error: {e}")
    # 统计预测结果中出现次数最多的类别
    predicted_genre = max(set(predictions), key=predictions.count)
    # 返回最终的预测类别
    return predicted_genre
#blues=4
# 要检测的音频文件路径
data_path = "D:\\smartimfor\\智能信息处理\\智能信息处理\\Myclassification\\Data\\test_audio"
genres = os.listdir(data_path)
total=0
right=0
wrong=0
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        try:
            y_data, sr = librosa.load(file_path, sr=None)
            total=total+1
        except Exception as e:
            # 如果遇到错误，打印警告并跳过该文件
            print(f"Warning: Failed to load {file_path}. Error: {e}\n")
            continue
        mel_array = preprocess_wav(file_path)
        slice_and_save_image(mel_array)
        # 进行音频切片并进行分类预测
        predicted_genre = predict_genre(mel_array, model)
        print(predicted_genre)
        if integer_to_genre(predicted_genre)==genre:
            right=right+1
        else :
            wrong=wrong+1
'''
file_path ="D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\genres_original\\classical\\classical.00000.wav"
# 预处理 WAV 文件
mel_array = preprocess_wav(file_path)
slice_and_save_image(mel_array)
# 进行音频切片并进行分类预测
predicted_genre = predict_genre(mel_array, model)
print("Predicted genre:", predicted_genre)
'''
print("预测样本总数=",total)
print("预测错误样本数=",wrong)
print("预测正确样本数=",right)
print("accuracy=",right/total)



