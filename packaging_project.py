import os
import shutil
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED, ZIP_LZMA

root_path = os.path.dirname(os.path.realpath(__file__))


def copytree(src_dir: str, dst_dir: str, exclude_filename: list = None):
    if exclude_filename is None:
        exclude_filename = []
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for fn in os.listdir(src_dir):
        if fn in exclude_filename:
            continue
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
        # print(os.path.realpath(dst))


def compress_file(file_path_list: list, output_path: str, compress_type: str = None) -> None:
    """
    将多个文件添加到压缩包。注意！若目标压缩包已存在，会被覆盖！
    :param file_path_list: list[str]. 需要压缩的文件路径列表，可以包含目录，但是不能包含磁盘根目录
    :param output_path: str. 输出文件路径，必须以.zip结尾
    :param compress_type: str. 默认为None。只接收None、"zip"、"lzma"，对应仅存储、zip算法压缩、lzma算法压缩模式
    :return: None
    """
    # 确定参数
    if compress_type is None:
        compress = ZIP_STORED
    elif compress_type == 'zip':
        compress = ZIP_DEFLATED
    elif compress_type == 'lzma':
        compress = ZIP_LZMA
    else:
        raise ValueError('compress_type参数只接收None、"zip"、"lzma"')

    # 创建压缩文件
    with ZipFile(output_path, 'w', compress) as z:
        # 遍历要压缩的文件路径
        for file_path in file_path_list:
            # 添加文件
            if os.path.isfile(file_path):
                z.write(file_path, os.path.basename(file_path))
                continue

            # 添加目录
            root = os.path.dirname(file_path)
            for father, dir_names, filenames in os.walk(file_path):
                rel_father = os.path.relpath(father, root)

                # 防止空目录不会被添加
                if not filenames:
                    z.write(
                        filename=father,
                        arcname=rel_father
                    )
                    continue

                # 添加非空目录的文件
                for filename in filenames:
                    z.write(
                        filename=os.path.join(father, filename),
                        arcname=os.path.join(rel_father, filename)
                    )

        z.close()


if __name__ == '__main__':
    package_name = 'code'
    package_path = 'C:/Users/Lenovo/Desktop/code'
    zip_path = '{}.zip'.format(package_path)
    os.makedirs(package_path, exist_ok=True)
    os.makedirs(os.path.join(package_path, 'checkpoints'), exist_ok=True)
    copytree(
        src_dir=os.path.join(root_path, 'dataset'),
        dst_dir=os.path.join(package_path, 'dataset'),
        exclude_filename=['train.csv', 'test.csv']
    )
    os.makedirs(os.path.join(package_path, 'eval_result'), exist_ok=True)
    filenames = [
        'compute_performance.py', 'config.py', 'data_preprocess.py', 'draw_graph.py', 'train.py', 'requirements.txt'
    ]
    for fn in filenames:
        shutil.copy(
            src=os.path.join(root_path, fn),
            dst=os.path.join(package_path, fn)
        )
    compress_file(
        file_path_list=[package_path],
        output_path=zip_path,
        compress_type='zip'
    )
    shutil.rmtree(package_path)
    print('已创建压缩包：{}'.format(os.path.realpath(zip_path)))
