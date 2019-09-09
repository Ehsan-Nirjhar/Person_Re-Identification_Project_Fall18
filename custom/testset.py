#################################################################################
######################## CSCE 625 : AI PROJECT : TEAM 17 ########################
## Prepares the validation dataset givan in the class
## Copy the 'testSet' folder to the '/data' folder
## Must contain:
##      /data/testSet/gallery/*.png   (all gallery images in .png format)
##      /data/testSet/query/*.png     (all query images in .png format)
#################################################################################
#################################################################################

import os.path as osp
from os import  listdir

class TestSetCSCE625 (object):
    """
    CSCE625 AI Project ValSet Dataset
    """
    dataset_dir = 'testSet'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir  = osp.join(root, self.dataset_dir)
        self.que_dir = osp.join(self.dataset_dir, 'query')
        self.gal_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        que, num_que_imgs = self._process_dir(self.que_dir)
        gal, num_gal_imgs = self._process_dir(self.gal_dir)
        num_tot_imgs = num_que_imgs + num_gal_imgs

        print("=> Test set for CSCE625 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   |  # images")
        print("  ------------------------------")
        print("  query    | {:8d}".format(num_que_imgs))
        print("  gallery  | {:8d}".format(num_gal_imgs))
        print("  ------------------------------")
        print("  total    | {:8d}".format(num_tot_imgs))
        print("  ------------------------------")

        self.query   = que
        self.gallery = gal

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.que_dir):
            raise RuntimeError("'{}' is not available".format(self.que_dir))
        if not osp.exists(self.gal_dir):
            raise RuntimeError("'{}' is not available".format(self.gal_dir))

    def _process_dir(self, dir_imgs):
        dataset = []
        
        all_files = listdir(dir_imgs)
        for file in all_files:
            if file.endswith('.png'):
                img_path = osp.join(dir_imgs,file)
                dataset.append((img_path, file[:-4], 0))

        num_imgs = len(dataset)
        return dataset, num_imgs
