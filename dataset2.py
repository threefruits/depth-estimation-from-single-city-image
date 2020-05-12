import torch
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from lu_vp_detect import VPDetection

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

length_thresh = 20
principal_point = None
focal_length = 256
seed = 1337
vpd = VPDetection(length_thresh, principal_point, focal_length, seed)


class CityDataset(Dataset):
    def __init__(self, tfms):
        super(CityDataset, self).__init__()
        self.images = np.load('img.npy')
        self.depth = np.load('depth.npy')
        # self.vp = np.load('vp.npy')
        self.tfms = tfms

    def __getitem__(self, index):
        fn_img = self.images[index]
        img = Image.open(fn_img)
        img = np.array(img, dtype = np.uint8)

        vps = vpd.find_vps(fn_img)
        # print(vps)
        Seg_line=vpd.create_debug_VP_image(show_image=False)
        

        # fn_vp = self.vp[index]
        # vps = np.load(fn_vp)
        # vps = vps['vpts']
        # img=np.transpose(img, axes=[2,1,0])
        # print(img.shape)
        # print(img)
        # img_ = Image.fromarray(img)
        # img_.show()
        # img_1 = Image.fromarray(img)
        # img_1.show()
       
        fn_dep = self.depth[index]
        dep = np.load(fn_dep)
        dep = dep['depth'].squeeze()
        # dep=np.transpose(dep, axes=[1,0])
        # print(dep)
        # dep = (dep/dep.max())*255
        # dep = dep*10
        dep[dep>100]=100
        dep[dep==0]=100
        dep = (dep/dep.max())*255
        dep = dep.astype(np.uint8)
        # print(dep.shape)
        # print(dep)
        # img_ = Image.fromarray(dep)
        # img_.show()
        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":dep})
            img, dep = tfmd_sample["image"], tfmd_sample["depth"]
 
        return (img,dep,vps,Seg_line)

    def __len__(self):
        return len(self.images)

class CityDataset_normal(Dataset):
    def __init__(self, tfms):
        super(CityDataset_normal, self).__init__()
        self.images = np.load('img.npy')
        self.normal = np.load('normal.npy')
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
       
        fn_normal = self.normal[index]
        normal = np.load(fn_normal)
        normal = normal['normal']
        # normal =np.transpose(normal, axes=[2,1,0])

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
        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":normal})
            img, dep = tfmd_sample["image"], tfmd_sample["depth"]
 
        return (img,dep)

    def __len__(self):
        return len(self.images)