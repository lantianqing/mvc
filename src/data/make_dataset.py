import os
import sys
import cv2
import numpy as np
import torch as th
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_blobs

import config


def export_dataset(name, views, labels):
    """导出数据集到npz文件
    
    Args:
        name: 数据集名称
        views: 视图列表
        labels: 标签数组
    """
    processed_dir = config.DATA_DIR / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    file_path = processed_dir / f"{name}.npz"
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)


def _concat_edge_image(img):
    """将边缘图像与原始图像连接
    
    Args:
        img: 原始图像
    
    Returns:
        连接后的图像
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return np.stack((img, edge), axis=-1)


def _mnist(add_edge_img, dataset_class=torchvision.datasets.MNIST, **dataset_kwargs):
    """加载MNIST数据集

    Args:
        add_edge_img: 是否添加边缘图像
        dataset_class: 数据集类
        **dataset_kwargs: 数据集的额外参数（如 EMNIST 的 split 参数）

    Returns:
        数据和标签
    """
    img_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if add_edge_img:
        img_transforms.insert(0, _concat_edge_image)
    transform = transforms.Compose(img_transforms)
    dataset = dataset_class(
        root=config.DATA_DIR / "raw",
        train=True,
        download=True,
        transform=transform,
        **dataset_kwargs
    )

    loader = th.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = list(loader)[0]
    return data, labels


def mnist_mv():
    """创建多视图MNIST数据集
    """
    data, labels = _mnist(add_edge_img=True)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("mnist_mv", views=views, labels=labels)


def fmnist():
    """创建多视图FashionMNIST数据集
    """
    data, labels = _mnist(add_edge_img=True, dataset_class=torchvision.datasets.FashionMNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("fmnist", views=views, labels=labels)


def emnist():
    """创建多视图EMNIST数据集（ByClass split，62个类：10数字+26大写+26小写）
    """
    data, labels = _mnist(add_edge_img=True, dataset_class=torchvision.datasets.EMNIST, split="byclass")
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("emnist", views=views, labels=labels)


def ccv():
    """创建CCV数据集
    """
    ccv_dir = config.DATA_DIR / "raw" / "CCV"

    def _load_train_test(typ, suffix="Feature"):
        """加载训练和测试数据
        
        Args:
            typ: 特征类型
            suffix: 文件后缀
        
        Returns:
            合并后的数据集
        """
        if typ:
            typ += "-"
        train = np.loadtxt(ccv_dir / f"{typ}train{suffix}.txt")
        test = np.loadtxt(ccv_dir / f"{typ}test{suffix}.txt")
        return np.concatenate((train, test), axis=0)

    views = [_load_train_test(typ) for typ in ["STIP", "SIFT", "MFCC"]]
    labels = _load_train_test("", suffix="Label")

    # 只包含恰好有一个标签的视频
    row_mask = (labels.sum(axis=1) == 1)
    labels = labels[row_mask].argmax(axis=1)
    views = [v[row_mask] for v in views]
    export_dataset("ccv", views=views, labels=labels)


def coil():
    """创建COIL数据集
    """
    from skimage.io import imread

    data_dir = config.DATA_DIR / "raw" / "COIL"
    img_size = (1, 128, 128)
    n_objs = 20  # 对象数量
    n_imgs = 72  # 每个对象的图像数量
    n_views = 3  # 视图数量
    assert n_imgs % n_views == 0

    n = (n_objs * n_imgs) // n_views  # 每个视图的样本数量

    imgs = np.empty((n_views, n, *img_size))
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        # 随机排列图像索引并 reshape 为视图数量
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            for i, idx in enumerate(indices):
                fname = data_dir / f"obj{obj + 1}__{idx}.png"
                img = imread(fname)[None, ...]
                imgs[view, ((obj * (n_imgs // n_views)) + i)] = img

    assert not np.isnan(imgs).any()
    views = [imgs[v] for v in range(n_views)]
    labels = np.array(labels)
    export_dataset("coil", views=views, labels=labels)


def blobs_overlap():
    """创建重叠的blobs数据集
    """
    nc = 1000  # 每个类的样本数量
    ndim = 2  # 特征维度
    view_1, l1 = make_blobs(n_samples=[nc, 2 * nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    view_2, l2 = make_blobs(n_samples=[2 * nc, nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    labels = l1 + l2
    export_dataset("blobs_overlap", views=[view_1, view_2], labels=labels)


def blobs_overlap_5():
    """创建5类重叠的blobs数据集
    """
    nc = 500  # 每个类的样本数量
    ndim = 2  # 特征维度
    view_1, _ = make_blobs(n_samples=[3 * nc, 2 * nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    view_2, _ = make_blobs(n_samples=[1 * nc, 2 * nc, 2 * nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    view_2[(2 * nc): (4 * nc)] = view_2[(2 * nc): (4 * nc)][::-1]
    labels = np.concatenate(([nc * [i] for i in range(5)]))
    export_dataset("blobs_overlap_5", views=[view_1, view_2], labels=labels)


# 数据集加载器字典
LOADERS = {
    "mnist_mv": mnist_mv,
    "ccv": ccv,
    "blobs_overlap": blobs_overlap,
    "blobs_overlap_5": blobs_overlap_5,
    "fmnist": fmnist,
    "coil": coil,
    "emnist": emnist
}


def main():
    """主函数，导出数据集
    """
    export_sets = sys.argv[1:] if len(sys.argv) > 1 else LOADERS.keys()
    for name in export_sets:
        print(f"导出数据集 '{name}'")
        LOADERS[name]()


if __name__ == '__main__':
    main()
