import numpy as np
import os


depth=[]
img=[]
sgmt=[]
plane=[]
normal=[]
vp=[]
path="./data/holicity"
files= os.listdir(path)
for p in files:
    path="./data/holicity/"
    files_month= os.listdir(path+p)
    files_month.sort(key=str.lower)
    for i in range( int(len(files_month)/12) ):
        depth.append(path+p+'/'+files_month[i*12+2])
        img.append(path+p+'/'+files_month[i*12+3])
        sgmt.append(path+p+'/'+files_month[i*12+9])
        plane.append(path+p+'/'+files_month[i*12+8])
	    # normal.append(path+p+'/'+files_month[i*12+5])
        normal.append(path+p+'/'+files_month[i*12+5])
        vp.append(path+p+'/'+files_month[i*12+11])
np.save( "depth.npy" ,depth )
np.save( "img.npy" ,img )
np.save( "sgmt.npy" ,sgmt )
np.save( "plane.npy", plane)
np.save( "normal.npy",normal)
np.save( "vp.npy",vp)
