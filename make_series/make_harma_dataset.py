import pdb
import shutil
import json
from pathlib import Path
import os
from tqdm import tqdm

def make_dir(data, origin_root, dest_root):

    os.makedirs(dest_root, exist_ok=True)

    for img in tqdm(data):
        # print(img)
        # pdb.set_trace()
        img_path = os.path.basename(img['image'])
        copy_image(src_path= origin_root + '/' + img_path,
                   dest_dir= dest_root)
    print('done')

def copy_image(
        src_path: str,
        dest_dir: str,
        preserve_metadata: bool = True
) -> bool:
    """
    复制图片文件到目标目录

    参数:
        src_path (str): 源图片路径（完整路径）
        dest_dir (str): 目标目录路径
        overwrite (bool): 是否覆盖已存在文件（默认False）
        preserve_metadata (bool): 是否保留文件元数据（默认True）

    返回:
        bool: 是否复制成功
    """
    try:
        # 转换为Path对象（兼容Windows/Linux路径）
        src = Path(src_path)
        dest_folder = Path(dest_dir)

        # 验证源文件是否存在
        if not src.exists():
            raise FileNotFoundError(f"源文件 {src_path} 不存在")
        if not src.is_file():
            raise ValueError(f"{src_path} 不是文件")

        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_file = dest_folder / src.name

        copy_func = shutil.copy2 if preserve_metadata else shutil.copy

        copy_func(str(src), str(dest_file))

        print(f"✅ 成功复制文件到：{dest_file}")
        return True

    except Exception as e:
        print(f"❌ 复制失败：{str(e)}")
        return False



if __name__ == "__main__":
    # 示例配置
    origin_root = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/RSITMD/images'

    destination_train = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/HarmaFinetune/rsitmd/images/train'
    destination_test = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/HarmaFinetune/rsitmd/images/test'
    destination_val = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/HarmaFinetune/rsitmd/images/val'

    train_json = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/HarmaFinetune/rsitmd_train.json'
    val_json = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/HarmaFinetune/rsitmd_val.json'
    test_json = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/HarmaFinetune/rsitmd_test.json'

    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    print(train_data[0].keys())
    print(train_data[0]['image'])

    print(test_data[0].keys())
    print(test_data[0]['image'])
    make_dir(train_data, origin_root, destination_train)
    make_dir(val_data, origin_root, destination_val)
    make_dir(test_data, origin_root, destination_test)