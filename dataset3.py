import torch
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
    # normalize
])


class CityDataset(Dataset):
    def __init__(self, tfms):
        super(CityDataset, self).__init__()
        self.images = np.load('img.npy')
        self.sgmt = np.load('sgmt.npy')
        self.tfms = tfms

    def __getitem__(self, index):
        fn_img = self.images[index]
        img = Image.open(fn_img)
        img = np.array(img, dtype = np.uint8)
        # img=np.transpose(img, axes=[2,1,0])
        # print(img.shape)
        # print(img)
        # img_ = Image.fromarray(img)
        # img_.show()
        # img_1 = Image.fromarray(img)
        # img_1.show()
        # print(img.shape)
        fn_sgmt = self.sgmt[index]
        sgmt = Image.open(fn_sgmt)
        sgmt=sgmt.resize((256,256))
        sgmt = np.array(sgmt, dtype = np.uint8)
        # sgmt2=np.zeros((128,128,5))
        
        # sgmt_c = sgmt.copy()
        # sgmt_c[sgmt_c != 0]=9
        # sgmt_c[sgmt_c == 0]=1
        # sgmt_c[sgmt_c == 9]=0
        # sgmt2[:,:,0]=sgmt_c

        # for i in range(4):
        #     sgmt_c = sgmt.copy()
        #     # print(sgmt_c)
        #     sgmt_c[sgmt_c != i+1]=0
        #     sgmt_c[sgmt_c == i+1]=1
        #     sgmt2[:,:,i+1]=sgmt_c
        # sgmt = sgmt2
        
        # print(sgmt[:,:,0])
        # print(sgmt.shape)
        # dep=np.transpose(dep, axes=[1,0])
        # print(dep)
        # dep = (dep/dep.max())*255
        # dep = dep*10
        # dep[dep>100]=100
        # dep[dep==0]=100
        # dep = (dep/dep.max())*255
        # dep = dep.astype(np.uint8)
        # print(dep.shape)
        # print(dep)
        # img_ = Image.fromarray(dep)
        # img_.show()
        # print(sgmt)
        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":sgmt})
            img, sgmt = tfmd_sample["image"], tfmd_sample["depth"]
            # img = self.tfms(img)
            # sgmt = preprocess(sgmt)
 
        return (img,sgmt)

    def __len__(self):
        return len(self.images)

class CityDataset_plane(Dataset):
    def __init__(self, tfms):
        super(CityDataset_plane, self).__init__()
        self.images = np.load('img.npy')
        self.plane = np.load('plane.npy')
        self.tfms = tfms

    def __getitem__(self, index):
        fn_img = self.images[index]
        img = Image.open(fn_img)
        img = np.array(img, dtype = np.uint8)
        # img=np.transpose(img, axes=[2,1,0])
        # print(img.shape)
        # print(img)
        # img_ = Image.fromarray(img)
        # img_.show()
        # img_1 = Image.fromarray(img)
        # img_1.show()
        # print(img.shape)
        fn_plane = self.plane[index]
        plane = Image.open(fn_plane)
        plane=plane.resize((256,256))
        plane = np.array(plane, dtype = np.uint8)

        plane_index=[]
        count=1
        for i in range(300):    
            m=plane[plane==i+2]
            if(m.size >400):
                count+=1
                plane[plane==i+2]=count
                # im[depth_tensor.cpu().numpy()==count]=np.array([107+50*i,27+104*i,167+38*i])
            else:
                plane[plane==i+2]=0
        # sgmt2=np.zeros((128,128,5))
        
        # sgmt_c = sgmt.copy()
        # sgmt_c[sgmt_c != 0]=9
        # sgmt_c[sgmt_c == 0]=1
        # sgmt_c[sgmt_c == 9]=0
        # sgmt2[:,:,0]=sgmt_c

        # for i in range(4):
        #     sgmt_c = sgmt.copy()
        #     # print(sgmt_c)
        #     sgmt_c[sgmt_c != i+1]=0
        #     sgmt_c[sgmt_c == i+1]=1
        #     sgmt2[:,:,i+1]=sgmt_c
        # sgmt = sgmt2
        
        # print(sgmt[:,:,0])
        # print(sgmt.shape)
        # dep=np.transpose(dep, axes=[1,0])
        # print(dep)
        # dep = (dep/dep.max())*255
        # dep = dep*10
        # dep[dep>100]=100
        # dep[dep==0]=100
        # dep = (dep/dep.max())*255
        # dep = dep.astype(np.uint8)
        # print(dep.shape)
        # print(dep)
        # img_ = Image.fromarray(dep)
        # img_.show()
        # print(sgmt)
        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":plane})
            img, plane = tfmd_sample["image"], tfmd_sample["depth"]
            # img = self.tfms(img)
            # sgmt = preprocess(sgmt)
 
        return (img,plane)

    def __len__(self):
        return len(self.images)