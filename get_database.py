import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import ipdb


class Dataset:
    def __init__(self, load_dir="/home/user/data/mat", mfilename="/KNEE_Rand6ACS10p.mat"):
        dirs           = sorted(os.listdir(load_dir))
        self.fnames    = []
        self.img_train = []
        self.img_valid = []
        self.img_test  = []
        self.img_shape = [320, 320, 8]
        start = time.time()
        cnt = 0
        for aDir in dirs:
            if os.path.splitext(aDir)[-1]==".mat":
                continue
            for aMat in sorted(os.listdir(load_dir+"/"+aDir)):
                if aDir in ["P17","P18"]:
                    self.img_valid.append(cnt)
                elif aDir in ["P19","P20"]:
                    self.img_test.append(cnt)
                else:
                    self.img_train.append(cnt)
 
                self.fnames.append(load_dir+"/"+aDir+"/"+aMat)
                cnt = cnt+1
        end = time.time()
        self.totalN = cnt
        # get mask
        m_1ch      = self.read_mat(load_dir+mfilename,"m")

        m_1ch      = np.fft.fftshift(m_1ch,(0,1))
        self.mask  = np.empty(self.img_shape)
        for i in range(self.img_shape[2]):
            self.mask[:,:,i]=m_1ch


    @staticmethod
    def read_mat(filename, var_name="img"):
        mat =  sio.loadmat(filename)
        return mat[var_name]
    
    def getBatch(self, IDstr, bStart, bEnd, doAug=0):
        if IDstr=="train":
            ids = self.img_train
        elif IDstr=="valid":
            ids = self.img_valid
        elif IDstr=="test":
            ids = self.img_test
        else:
            ids=[]
            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        bEnd  = min(bEnd, len(ids))
        cur_ids = ids[bStart:bEnd]
        
        batch_num = bEnd-bStart
        nCh       = self.img_shape[2]
        targets   = np.empty([batch_num, self.img_shape[0], self.img_shape[1], nCh*2 ], dtype=np.float32)
        inputs    = np.empty([batch_num, self.img_shape[0], self.img_shape[1], nCh*2 ], dtype=np.float32)

        for iB, cur_id in enumerate(cur_ids):
            orig_img               = self.read_mat(self.fnames[cur_id])
            orig_k                 = np.fft.fft2(orig_img, axes=(0,1))
            down_k                 = np.multiply(orig_k, self.mask)
            down_img               = np.fft.ifft2(down_k, axes=(0,1))
            inputs[iB,:,:,0:nCh]   = np.real(down_img)
            inputs[iB,:,:,nCh:]  = np.imag(down_img)
            targets[iB,:,:,0:nCh]  = np.real(orig_img)
            targets[iB,:,:,nCh:] = np.imag(orig_img)
        return  inputs, targets

    def shuffleTrainIDs(self):
        random.seed(0)
        random.shuffle(self.img_train)

def plot(x_complex):
    abs_x = np.abs(x_complex)
    plt.imshow(abs_x[:,:,0])
    plt.show()


   

if __name__=="__main__":
    MRDB = Dataset()
