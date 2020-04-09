from tqdm import tqdm
import os
import time

def get_neg(fam1,fiwpath):
    import random
    random.seed(time.time())
    famlist=os.listdir(fiwpath)
    fam3=random.choice(famlist)
    fampath=fiwpath+"/"+fam3
    while fam3 ==fam1 or not os.path.isdir(fampath) or int(fam3[1:])>=900:
        fam3=random.choice(famlist)
        fampath=fiwpath+"/"+fam3
    memlist=os.listdir(fampath)
    mem3=random.choice(memlist)
    mempath=fampath+"/"+mem3
    if not os.path.isdir(mempath):
        return get_neg(fam1,fiwpath)
         
    if os.path.isdir(mempath):
        filelist=os.listdir(mempath)
        if(len(filelist)<=0):
            return get_neg(fam1,fiwpath)
        file3=random.choice(filelist)
        img3=fam3+"/"+mem3+"/"+file3
        return img3
    else:
        return get_neg(fam1)

def get_csv(inpath,outpath,fiwpath):
    f_in = open(inpath, "r")
    f_out = open(outpath, "w")
    lines=f_in.readlines()

    for line in tqdm(lines):
        label,img1,img2=line[:-1].split(",")
        if(label=="1"):
            fam1,mem1,file1=img1.split("/")
            img3=get_neg(fam1,fiwpath)
            outline=img1+","+img2+","+img3+"\n"
            f_out.write(outline)
    f_in.close()
    f_out.close()
