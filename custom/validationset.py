#################################################################################
######################## CSCE 625 : AI PROJECT : TEAM 17 ########################
## Prepares the validation dataset givan in the class
## Copy the 'valSet' folder to the '/data' folder
## Must contain:
##      /data/valSet/gallery/*.png   (all gallery images in .png format)
##      /data/valSet/query/*.png     (all query images in .png format)
##      /data/valSet/galleryInfo.txt
##      /data/valSet/queryInfo.txt
#################################################################################
#################################################################################

import os.path as osp

class ValSetCSCE625 (object):
    """
    CSCE625 AI Project ValSet Dataset
    """
    dataset_dir = 'valSet'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir  = osp.join(root, self.dataset_dir)
        self.que_dir = osp.join(self.dataset_dir, 'query')
        self.gal_dir = osp.join(self.dataset_dir, 'gallery')
        self.que_inf = osp.join(self.dataset_dir, 'queryInfo.txt')
        self.gal_inf = osp.join(self.dataset_dir, 'galleryInfo.txt')

        self._check_before_run()

        que, num_que_pids, num_que_imgs = self._process_dir(self.que_dir, self.que_inf)
        gal, num_gal_pids, num_gal_imgs = self._process_dir(self.gal_dir, self.gal_inf)
        num_tot_pids = num_que_pids
        num_tot_imgs = num_que_imgs + num_gal_imgs

        print("=> Validation set for CSCE625 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(num_que_pids, num_que_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gal_pids, num_gal_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_tot_pids, num_tot_imgs))
        print("  ------------------------------")

        self.query   = que
        self.gallery = gal

        self.num_query_pids   = num_que_pids
        self.num_gallery_pids = num_gal_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.que_dir):
            raise RuntimeError("'{}' is not available".format(self.que_dir))
        if not osp.exists(self.gal_dir):
            raise RuntimeError("'{}' is not available".format(self.gal_dir))

    def _process_dir(self, dir_imgs, dir_info):
        pid_container = set()
        dataset = []
        with open(dir_info, 'r') as f:
            for x in f.readlines():
                img, pid = x.split('\t')
                img_path = osp.join(dir_imgs, img+'.png')
                pid = int(pid)
                pid_container.add(pid)
                camid = 0
                dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
