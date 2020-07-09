import glob
import re

import os
import os.path as osp
import numpy as np

from PIL import Image

from .bases import BaseImageDataset


class CUB200(BaseImageDataset):

    dataset_dir = "CUB_200_2011"

    def __init__(self, root, verbose=True, **kwargs):
        super(CUB200, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        # query and gallery are same for the image retrieval task
        self.query_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir)
        gallery = query

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, _ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, _ = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'/([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for idx, img_path in enumerate(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, idx))

        return dataset


def split_dataset(dataset_root):
    """
    Args:
        dataset_root: The root path of original CUB200-2011 dataset. It contains `attributes`, `images`, etc.

    Returns:

    """
    bounding_boxes = np.genfromtxt(osp.join(dataset_root, "bounding_boxes.txt"))
    images = np.genfromtxt(osp.join(dataset_root, "images.txt"), dtype=str)

    train_path = osp.join(dataset_root, "bounding_box_train")
    test_path = osp.join(dataset_root, "bounding_box_test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)


    pattern = re.compile(r"^([\d]+)\.[\w]+/[\w]+([\d]+)_([\d]+)")
    for bounding_box, image_name in zip(bounding_boxes, images):
        image_id, x0, y0, w, h = bounding_box
        _, image_name = image_name

        class_id, first, second = map(int, pattern.search(image_name).groups())

        image = Image.open(osp.join(dataset_root, "images", image_name))
        crop_image = image.crop((x0, y0, x0 + w, y0 + h))

        class_num = int(image_name.split(".")[0])
        if class_num > 100:
            output_path = test_path
        else:
            output_path = train_path

        crop_image.save(osp.join(output_path, f"{class_id:03}_{first:04}_{second:06}.jpg"))


if __name__ == "__main__":
    split_dataset("/mnt/nas59_data/Image_retrieval/CUB_200_2011")