import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
import os
import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import scipy.io as scio
import mat73

def np_range_norm(image, maxminnormal=True, range1=True):

    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if maxminnormal:
            _min = image.min()
            _range = image.max() - image.min()
            narmal_image = (image - _min) / _range
            if range1:
               narmal_image = (narmal_image - 0.5) * 2
        else:
            _mean = image.mean()
            _std = image.std()
            narmal_image = (image - _mean) / _std

    return narmal_image



class ReconDataset(data.Dataset):
    __inputdata = []
    __inputimg = []
    __outputdata = []
    a = []
    b = []
    c = []

    def __init__(self,root, train=True, das=True,transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__inputimg = []

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "train_3D.mat"  # load train dataset
            matdata = mat73.loadmat(folder)
            # train
            for i in range(250):

                a = matdata['signal_train'][:, :,:,i]
                c = matdata['image_train'][0:2,:, :,:,i]  #Select the first three channels in the label:hbo,hb,water
                b = matdata['mri_train'][:, :, :,i]
                self.__inputdata.append(a[np.newaxis, :, :,:])
                self.__inputimg.append(b[np.newaxis, :, :, :])
                self.__outputdata.append(c)
        else:
            folder = root + "test.mat"  #********** Modify the input file name ********#
            matdata = mat73.loadmat(folder)
            # tset
            a = matdata['signal_real'][ :, :, :]
            b = matdata['mri_real'][:, :, :]
            self.__inputdata.append(a[np.newaxis,:, :])
            self.__inputimg.append(b[np.newaxis, :, :])




    def __getitem__(self, index):
      
        

        rawdata =  self.__inputdata[index] #.reshape((1,1,2560,120))
        # reconstruction =self.__outputdata[index] #.reshape((1,1,2560,120))
        beamform = self.__inputimg[index]


        rawdata = torch.Tensor(rawdata)
        # reconstructions = torch.Tensor(reconstruction)
        beamform = torch.Tensor(beamform)

        # return rawdata, reconstructions,beamform
        return rawdata, beamform

    def __len__(self):
        return len(self.__inputdata)





if __name__ == "__main__":
    dataset_pathr = 'D:/LSM/3D-ZNet/'

    mydataset = ReconDataset(dataset_pathr,train=False,das=True)
    #print(mydataset.__getitem__(3))
    train_loader = DataLoader(
        mydataset,
        batch_size=1, shuffle=True)
    batch_idx, (rawdata, reimage) = list(enumerate(train_loader))[0]


    print('raw:',rawdata.size())
    print('reimage:', reimage.size())
    reimage = reimage[:,:,:,:,70].squeeze()
    plt.imshow(reimage)
    plt.show()
    # print('bfimg:', bfimg.size())
    # print(rawdata.max())
    # print(rawdata.min())
    # print(mydataset.__len__())






