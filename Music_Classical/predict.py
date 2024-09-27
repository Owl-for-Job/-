import os
import shutil


def convert_image_path_to_audio_path(image_path):
    # 拆分路径
    path_parts = image_path.split(os.sep)

    # 获取类别（如'blues'）
    genre = path_parts[-2]

    # 获取编号（如'00000'）
    filename = path_parts[-1]
    file_number = filename.split('_')[0]

    # 构建新的路径
    base_path = os.sep.join(path_parts[:-4])  # 获取到 'D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data'
    audio_path = os.path.join(base_path, 'genres_original', genre, f'{genre}.{file_number}.wav')

    return audio_path


def copy_audio_file(image_path, destination_base):
    # 获取音频文件路径
    audio_path = convert_image_path_to_audio_path(image_path)
    print(audio_path)
    # 获取类别（如'blues'）
    path_parts = image_path.split(os.sep)
    genre = path_parts[-2]

    # 创建目标目录路径
    destination_dir = os.path.join(destination_base, genre)

    # 确保目标目录存在
    os.makedirs(destination_dir, exist_ok=True)

    # 目标文件路径
    destination_path = os.path.join(destination_dir)

    # 使用 shutil.copyfile() 复制文件
    shutil.copy(audio_path, destination_path)
    print(f"文件已复制到 {destination_path}")


# 示例路径
dir_path = r"D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\Slice_Image\test"
destination_base = r"D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\test_audio"
genres= os.listdir(dir_path)
for genre in genres:
    genre_path = os.path.join(dir_path, genre)
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        copy_audio_file(file_path,destination_base)

