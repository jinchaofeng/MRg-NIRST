import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer


from mydataset_real import ReconDataset
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from skimage.measure import compare_ssim, compare_psnr
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
test_batch = 1
dataset_pathr = '../input/' # Dataset path:...\reconstruction\input


test_dataset = ReconDataset(dataset_pathr,train=False, das=True)


def normalization(data):

    _range = np.max(data) - np.min(data)

    return (data - np.min(data)) / _range
def data_norm(df,*cols):
    df_n = df.copy()
    for col in cols:
        ma = df_n[col].max()
        mi = df_n[col].min()
        df_n[col + '_n'] = (df_n[col] - mi) / (ma - mi)
    return(df_n)


print("loading data")
start1 = timer()

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch,
        shuffle=False)


for batch_idx, (rawdata,  bfimg) in enumerate(test_loader):
    rawdata = rawdata.to(device)
    # reimage = reimage.to(device)
    bfimg = bfimg.to(device)



model = torch.load('../model/Znet_3d.pkl',map_location='cuda:0')  # Model path:...\reconstruction\model


# print(model)

model = model.to(device)
print("got net")


# model.load_weights(dataset_pathr + 'Ynet.pkl')
print('predict test data')



outputs = model(rawdata, bfimg)
outputs = outputs.cuda().data.cpu().numpy()
# outputs =normalization(outputs)
# reimage= reimage.cuda().data.cpu().numpy()
bfimg= bfimg.cuda().data.cpu().numpy()
rawdata= rawdata.cuda().data.cpu().numpy()
outputs = outputs.squeeze()




print('done')
dataset_pathr1 = '../output/'  # Output path:...\reconstruction\output
sio.savemat(dataset_pathr1 + '/out_test_01.mat', {'val_output': outputs})   # *********Modify the output file name **********/
# print(torch.__version__)
time = round(timer() - start1, 2)
print(time)



