## CSCE 625 Fall 2018: TEAM 17
### Members:
1. Nipun Nath
2. Ehsanul Haque Nirjhar
3. Jennifer Yu



**Github repository used:**
- https://github.com/michuanhaohao/AlignedReID
- https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch
- https://github.com/yokattame/SpindleNet


In this project, we tried to improve the performance of AlignedReID by incorporating local features obtained from vertical strips. Detailed will be given in the report.


# Dataset Preparation

We used Market1501 and DukeMTMC-reID datasets for training purpose. We trained with individual datasets, as well as the combined datasets. 
Combination of training splits of the 2 datasets is done using the codes from 2nd repository.

## Market1501

Download the Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```


## DukeMTMC-reID

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1P9Jr0en0HBu_cZ7txrb2ZA_dI36wzXbS) or [BaiduYun](https://pan.baidu.com/s/1miIdEek). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```


## Combining Training Set of Market1501 and DukeMTMC-reID

If combined datasets are to be used for training, transformation scripts are to be run. You can find the combined data in https://drive.google.com/drive/folders/1hmZIRkaLvLb_lA1CcC4uGxmA4ppxPinj


Run the following script to transform the Market1501 dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

Run the following script to transform the DukeMTMC dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```

Finally combine these two datasets using the following script-

```bash
python script/dataset/combine_trainval_sets.py \
--market1501_im_dir ~/Dataset/market1501/images \
--market1501_partition_file ~/Dataset/market1501/partitions.pkl \
--duke_im_dir ~/Dataset/duke/images \
--duke_partition_file ~/Dataset/duke/partitions.pkl \
--save_dir ~/Dataset/market1501_duke
```


# Training

Run the following script-

```bash
python train_alignedreid.py  -d [dataset] -a resnet50
```

You can use `market1501`, `dukemtmcreid` or `market1501_duke` in the dataset field. By default, maximum epoch is set to 400 and printing frequency is set to 10. 

# Testing




## Current model:
* **Model**: AlignedReID

* **Weight**: [Google Drive Weight]

* **Performance**: mAP=79.1%, Rank-1: 91.8% on Market1501 dataset

## Current model's performance on the "valSet" (validation set):

* mAP: 79.0%

*  Rank-1  : 73.0%,
Rank-5  : 85.1%,
Rank-10 : 91.5%,
Rank-20 : 96.2%,

All calculated features and distance matrix for the valSet can be found here: [Google Drive valSet Result]

Interpretation:
~~~~
result = np.load('valSet_results.npy).item()
~~~~
`result['distmat']` is a 150x500 **distance matrix** for 150 query images and 500 gallery images.

`result['qf']` is a 150x2048 matrix containing 2048 **global features** for each of the 150 **query images**.

`result['gf']` is a 500x2048 matrix containing 2048 **global features** for each of the 500 **gallery images**.

`result['lqf']` is a 150x8x2048 matrix containing 2048 **local features** for 8 **horizontal stripes** for each of the 150 **query images**.

`result['lgf']` is a 500x8x2048 matrix containing 2048 **global features** for 8 **horizontal stripes**  for each of the 500 **gallery images**.

`result['q_pids']` IDs for the query images

`result['g_pids']` IDs for the gallery images

## How it works?

Come to our presentation (TBD) for more info. Here is a **sneak peek** how aligned distances are calculated:

![alt text](https://raw.githubusercontent.com/michuanhaohao/AlignedReID/master/imgs/Figure_0.png "Logo Title Text 1")



[Google Drive Weight]: https://drive.google.com/open?id=12ZVY23XMcThNsglhRfLYFr566-fM1cR2

[Google Drive valSet Result]: https://drive.google.com/open?id=1oT2b43yQoxh9hITYI1mIKvw3dHMICTN8
