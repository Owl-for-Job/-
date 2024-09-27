import os
import shutil
from pathlib import Path
import random

# 定义源和目标目录
source_dir = Path(r"D:\smartimfor\智能信息处理\智能信息处理\Myclassification\Data\genres_original")
target_base_dir = Path(r"E:\DataSet\智能信息处理\au_data")
target_dirs = [target_base_dir / "_train", target_base_dir / "_validation", target_base_dir / "_test"]

# 确保目标子文件夹存在，并清空它们
for target_dir in target_dirs:
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    for file in target_dir.iterdir():
        if file.is_file():
            file.unlink()

# 获取所有源文件夹
subfolders = os.listdir(source_dir)

# 定义一个函数来拷贝文件
def copy_files(files, target_dir):
    for file in files:
        shutil.copy(file, target_dir)


# 遍历每个子文件夹
for subfolder in subfolders:
    path=Path(os.path.join(source_dir,subfolder))
    # 获取子文件夹中的所有文件
    all_files = list(path.glob("*.wav"))

    # 随机打乱文件顺序
    random.shuffle(all_files)

    # 分配文件到目标文件夹，按8:1:1的比例
    total_files = len(all_files)
    train_split = int(total_files * 0.8)
    val_split = int(total_files * 0.1)

    train_files = all_files[:train_split]
    val_files = all_files[train_split:train_split + val_split]
    test_files = all_files[train_split + val_split:]

    # 拷贝文件到对应的文件夹
    for target_dir in target_dirs:
        class_path=target_dir/subfolder
        if not class_path.exists():
            class_path.mkdir(parents=True,exist_ok=True)

    copy_files(train_files, target_dirs[0]/subfolder)
    copy_files(val_files, target_dirs[1]/subfolder)
    copy_files(test_files, target_dirs[2]/subfolder)

print("文件已成功分配到目标文件夹。")
