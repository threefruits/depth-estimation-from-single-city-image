import numpy as np
import os


depth=[]
img=[]
path="./data/holicity"
files= os.listdir(path)
for p in files:
    path="./data/holicity/"
    files_month= os.listdir(path+p)
    files_month.sort(key=str.lower)
    for i in range( int(len(files_month)/12) ):
        depth.append(path+p+'/'+files_month[i*12+2])
        img.append(path+p+'/'+files_month[i*12+3])

np.save( "depth.npy" ,depth )
np.save( "img.npy" ,img )

