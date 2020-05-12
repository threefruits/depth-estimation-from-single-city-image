import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def plot_image(tup):
    img_tensor, depth_tensor = tup
    fig, axes = plt.subplots(1, 2, figsize=(10,15))
    for i,ax in enumerate(axes.flat):
        if(i==0):
            plot_image_tensor_in_subplot(ax, img_tensor)
        else:
            plot_depth_tensor_in_subplot(ax, depth_tensor)
        hide_subplot_axes(ax)

    plt.tight_layout()


def plot_image_sgmt(tup):
    img_tensor, depth_tensor = tup
    fig, axes = plt.subplots(1, 2, figsize=(10,15))
    for i,ax in enumerate(axes.flat):
        if(i==0):
            plot_image_tensor_in_subplot(ax, img_tensor)
        else:
            plot_sgmt_tensor_in_subplot(ax, depth_tensor)
        hide_subplot_axes(ax)

    plt.tight_layout()

def plot_image_normal(tup):
    img_tensor, depth_tensor = tup
    fig, axes = plt.subplots(1, 2, figsize=(10,15))
    for i,ax in enumerate(axes.flat):
        if(i==0):
            plot_image_tensor_in_subplot(ax, img_tensor)
        else:
            plot_normal_tensor_in_subplot(ax, depth_tensor)
        hide_subplot_axes(ax)

    plt.tight_layout()

def plot_image_plane(tup):
    img_tensor, plane_tensor = tup
    fig, axes = plt.subplots(1, 2, figsize=(10,15))
    for i,ax in enumerate(axes.flat):
        if(i==0):
            plot_image_tensor_in_subplot(ax, img_tensor)
        else:
            plot_plane_tensor_in_subplot(ax, plane_tensor)
        hide_subplot_axes(ax)

    plt.tight_layout()

#subplot utils    
def hide_subplot_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_image_tensor_in_subplot(ax, img_tensor):
    im = img_tensor.cpu().numpy().transpose((1,2,0))
    #pil_im = Image.fromarray(im, 'RGB')
    ax.imshow(im)

def plot_sgmt_tensor_in_subplot(ax, depth_tensor):
    im = np.zeros((256,256,3),dtype=np.uint8)
   
    im[depth_tensor.cpu().numpy()==0]=np.array([23,190,207])
    im[depth_tensor.cpu().numpy()==1]=np.array([148,103,189])
    im[depth_tensor.cpu().numpy()==3]=np.array([127,127,127])
    im[depth_tensor.cpu().numpy()==4]=np.array([44,160,44])
    # print(im)
    #im = im*255
    #im = im.astype(np.uint8)
    #pil_im = Image.fromarray(im, 'L')
    ax.imshow(im)


def plot_normal_tensor_in_subplot(ax, depth_tensor):
    im = depth_tensor.cpu().numpy().transpose((1,2,0))
    im = im*255
    #im = im*255
    im = im.astype(np.uint8)
    #pil_im = Image.fromarray(im, 'L')
    ax.imshow(im)


def plot_plane_tensor_in_subplot(ax, depth_tensor):
    im = np.zeros((256,256,3),dtype=np.uint8)
    # plane_index=[]
    # count=1
    
    # for i in range(250):    
    #     m=depth_tensor.cpu().numpy()[depth_tensor.cpu().numpy()==i+2]
    #     if(m.size >400):
    #         count+=1
    #         depth_tensor.cpu().numpy()[depth_tensor.cpu().numpy()==i+2]=count
            
    #     else:
    #         depth_tensor.cpu().numpy()[depth_tensor.cpu().numpy()==i+2]=0
    
    im[depth_tensor.cpu().numpy()==0]=np.array([23,190,207])
    im[depth_tensor.cpu().numpy()==1]=np.array([148,103,189])
    for i in range(16):
        im[depth_tensor.cpu().numpy()==i+2]=np.array([18+36*i,63+62*i,189+102*i])
    # im[depth_tensor.cpu().numpy()==4]=np.array([44,160,44])
    # print(im)
    #im = im*255
    #im = im.astype(np.uint8)
    #pil_im = Image.fromarray(im, 'L')
    ax.imshow(im)


def plot_depth_tensor_in_subplot(ax, depth_tensor):
    im = depth_tensor.cpu().numpy()
    # im = depth_tensor.cpu().detach().numpy().squeeze()
    #im = im*255
    #im = im.astype(np.uint8)
    #pil_im = Image.fromarray(im, 'L')
    ax.imshow(im,'gray')
    
def plot_model_predictions_on_sample_batch(images, depths, preds, plot_from=0, figsize=(12,12)):
    n_items=2
    fig, axes = plt.subplots(n_items, 3, figsize=figsize)
    
    for i in range(n_items):
        plot_image_tensor_in_subplot(axes[i,0], images[plot_from+i])
        plot_depth_tensor_in_subplot(axes[i,1], depths[plot_from+i])
        plot_depth_tensor_in_subplot(axes[i,2], preds[plot_from+i])
        hide_subplot_axes(axes[i,0])
        hide_subplot_axes(axes[i,1])
        hide_subplot_axes(axes[i,2])
    
    plt.tight_layout()

def plot_model_predictions_on_sample_batch_2(images, depths, preds, plot_from=0, figsize=(12,12)):
    n_items=5
    fig, axes = plt.subplots(n_items, 3, figsize=figsize)
    
    for i in range(n_items):
        plot_image_tensor_in_subplot(axes[i,0], images[plot_from+i])
        plot_sgmt_tensor_in_subplot(axes[i,1], depths[plot_from+i])
        plot_sgmt_tensor_in_subplot(axes[i,2], preds[plot_from+i])
        hide_subplot_axes(axes[i,0])
        hide_subplot_axes(axes[i,1])
        hide_subplot_axes(axes[i,2])
    
    plt.tight_layout()

def plot_model_predictions_on_sample_batch_3(images, depths, preds, plot_from=0, figsize=(12,12)):
    n_items=1
    fig, axes = plt.subplots(n_items, 3, figsize=figsize)
    
    for i in range(n_items):
        plot_image_tensor_in_subplot(axes[i,0], images[plot_from+i])
        plot_normal_tensor_in_subplot(axes[i,1], depths[plot_from+i])
        plot_normal_tensor_in_subplot(axes[i,2], preds[plot_from+i])
        hide_subplot_axes(axes[i,0])
        hide_subplot_axes(axes[i,1])
        hide_subplot_axes(axes[i,2])
    
    plt.tight_layout()