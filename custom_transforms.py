import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

class ResizeImgAndDepth(object):
    def __init__(self, size_tup):
        self.size = size_tup
        
    def __call__(self, sample):
        img = Image.fromarray(sample['image'],  'RGB').resize(self.size)
        # depth = Image.fromarray(sample['depth'], 'L').resize((self.size[0]//2, self.size[1]//2))
        depth = Image.fromarray(sample['depth'], 'L').resize(self.size)
        return { 'image' : np.array(img), 'depth' : np.array(depth) }

class ResizeImgAndNormal(object):
    def __init__(self, size_tup):
        self.size = size_tup
        
    def __call__(self, sample):
        img = Image.fromarray(sample['image'],  'RGB').resize(self.size)
        # depth = Image.fromarray(sample['depth'], 'L').resize((self.size[0]//2, self.size[1]//2))
        # print(sample['image'].shape)
        # depth = Image.fromarray(sample['depth'], 'RGB').resize(self.size)
        # print( np.array(depth))
        return { 'image' : np.array(img), 'depth' : sample['depth'] }


class ResizeImgAndSgmt(object):
    def __init__(self, size_tup):
        self.size = size_tup
        
    def __call__(self, sample):
        img = Image.fromarray(sample['image'],  'RGB').resize(self.size)
        # depth = Image.fromarray(sample['depth'], 'L').resize((self.size[0]//2, self.size[1]//2))
        # print(sample['depth'])
        return { 'image' : np.array(img), 'depth' : sample['depth'] }

class RandomHorizontalFlip(object):    
    def __call__(self, sample):
        img = sample["image"]
        depth = sample["depth"]
        # print(sample['depth'])
        if np.random.random() > 0.5:
            img = np.fliplr(sample['image']).copy()
            depth = np.fliplr(sample['depth']).copy()
        return { 'image' : img, 'depth' : depth }
       
class ImgAndDepthToTensor(object):        
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
        
    def __call__(self, sample):        
        return { 'image' : self.ToTensor(sample['image']), 'depth' : torch.tensor(sample['depth'], dtype=torch.float) }      
        # return { 'image' : self.ToTensor(sample['image']), 'depth' : self.ToTensor(sample['depth']) }       

class ImgAndNormalToTensor(object):        
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
        
    def __call__(self, sample):        
        # return { 'image' : self.ToTensor(sample['image']), 'depth' : torch.tensor(sample['depth'], dtype=torch.float) }      
        return { 'image' : self.ToTensor(sample['image']), 'depth' : self.ToTensor(sample['depth']) }    

class ImgAndSgmtToTensor(object):        
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
        
    def __call__(self, sample):        
        # print(sample['depth'])
        return { 'image' : self.ToTensor(sample['image']), 'depth' : torch.tensor(sample['depth'], dtype=torch.long) }      
        # return { 'image' : self.ToTensor(sample['image']), 'depth' : self.ToTensor(sample['depth']) }    

class NormalizeImg(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, sample):

        return { 'image' : self.normalize(sample['image']), 'depth' : sample['depth'] }

class UnNormalizeImgBatch(object):
    def __init__(self, mean, std):
        self.mean = mean.reshape((1,3,1,1))
        self.std = std.reshape((1,3,1,1))
    
    def __call__(self, batch):
        return (batch*self.std) + self.mean
