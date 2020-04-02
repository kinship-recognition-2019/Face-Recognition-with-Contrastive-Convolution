from tqdm import tqdm
import os
import time
inpath="fs_test.csv"
outpath="fs_test_triples.csv"
fiwpath="./FIDs_NEW"

f_in=open(inpath,"r")
f_out=open(outpath,"w")

def get_neg(fam1):
    import random
    random.seed(time.time())
    famlist=os.listdir(fiwpath)
    fam3=random.choice(famlist)
    fampath=fiwpath+"/"+fam3
    while fam3 ==fam1 or not os.path.isdir(fampath):
        fam3=random.choice(famlist)
        fampath=fiwpath+"/"+fam3
    memlist=os.listdir(fampath)
    mem3=random.choice(memlist)
    mempath=fampath+"/"+mem3
    if not os.path.isdir(mempath):
        return get_neg(fam1)
         
    if os.path.isdir(mempath):
        filelist=os.listdir(mempath)
        if(len(filelist)<=0):
            return get_neg(fam1)
        file3=random.choice(filelist)
        img3=fam3+"/"+mem3+"/"+file3
        return img3
    else:
        return get_neg(fam1)
        

lines=f_in.readlines()

for line in tqdm(lines):
    label,img1,img2=line[:-1].split(",")
    if(label=="1"):
        fam1,mem1,file1=img1.split("/")
        fam2,mem2,file2=img2.split("/")
        img3=get_neg(fam1)
        outline=img1+","+img2+","+img3+"\n"
        f_out.write(outline)
