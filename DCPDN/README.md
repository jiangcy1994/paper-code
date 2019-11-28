Densely Connected Pyramid Dehazing Network

arxiv.org/abs/1803.08396

CVPR2017

dataset:

Training images: https://drive.google.com/drive/folders/1Qv7SIZBVAtb9G1d6iVKu_8rVSsXJdv26?usp=sharing

Testing images: https://drive.google.com/drive/folders/1q5bRQGgS8SFEGqMwrLlku4Ad-0Tn3va7?usp=sharing

Testing images: https://drive.google.com/drive/folders/1hbwYCzoI3R3o2Gj_kfT6GHG7RmYEOA-P?usp=sharing

All the samples (both training and testing) are strored in Hdf5 file. You can also generate your sample using 'create_train.py' (Please download the NYU-depth @ http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)

Following are the sample python codes how to read the Hdf5 file:

```
file_name=self.root+'/'+str(index)+'.h5'
f=h5py.File(file_name,'r')

haze_image=f['haze'][:]
gt_trans_map=f['trans'][:]
gt_ato_map=f['ato'][:]
GT=f['gt'][:]
```
