import pandas as pd
import os
import cv2

#uncomment the necessary lines to do preprocessing of the data
#strr="artist";
#strr="genre";
strr="style";

orgdata = "./wikiart"
dir = "./wikiart_csv"
dest_path = "./data"
#print(os.walk(dir))
tflag=0;
vflag=0;
vcnt = 1;
tcnt = 1;
for root, folders, files in os.walk(dir):
        for filename in files:
            if strr in filename:
                print("in looop")
                if "class" in filename:
                    name_path = os.path.join(root,filename)
                    x = pd.read_csv(name_path)
                    dest_path = dest_path+ strr
                    ub = x.name.unique() 
                    if not os.path.isdir(dest_path+"train"): 
                        os.mkdir(dest_path+"train")  
                        for u in ub:
                            os.mkdir(os.path.join(dest_path+"train",'$'+u.split(' ')[0]+'$'+u.split(' ')[-1]))
                    if not os.path.isdir(dest_path+"val"): 
                        os.mkdir(dest_path+"val")  
                        for u in ub:
                            os.mkdir(os.path.join(dest_path+"val",'$'+u.split(' ')[0]+'$'+u.split(' ')[-1]))
                if "train" in filename:
                    print("preprocessing training dataset now\n")
                    name_path = os.path.join(dir,filename)
                    x = pd.read_csv(name_path)
                    itr = x.iterrows()
                    if tflag==0:
                        for i,row in itr:
                            classlabel = str(row['class'])
                            imgpath = row['fname']
                            #print(orgdata+'/'+imgpath.replace("/", "\\"))
                            image = cv2.imread(orgdata+'/'+imgpath.replace("/", "\\"))
                            resizedImg  = cv2.resize(image, (224, 224))
                            for root, folders, files in os.walk(dest_path+"train"):
                                for foldername in folders:
                                    if '$'+classlabel+'$' in foldername:
                                        writepath = os.path.join(root,foldername,imgpath.split('/')[-1])
                                        #print(writepath)
                                        cv2.imwrite(writepath, resizedImg)
                                        print("trainnnnn\n", tcnt)
                                        tcnt=tcnt+1
                        tflag = 1
                if "val" in filename:
                    print("preprocessing validation dataset now\n")
                    name_path = os.path.join(dir,filename)
                    #print(name_path)
                    x = pd.read_csv(name_path)
                    itr = x.iterrows()
                    if vflag==0:
                        for i,row in itr:
                            classlabel = str(row['class'])
                            imgpath = row['fname']
                            image = cv2.imread(orgdata+'/'+imgpath.replace("/", "\\"))
                            resizedImg  = cv2.resize(image, (224, 224))
                            for root, folders, files in os.walk(dest_path+"val"):
                                for foldername in folders:
                                    if '$'+classlabel+'$' in foldername:
                                        #print(foldername)
                                        writepath = os.path.join(root,foldername,imgpath.split('/')[-1])
                                        #print(writepath)
                                        cv2.imwrite(writepath, resizedImg)
                                        print("validdddd\n", vcnt)
                                        vcnt=vcnt+1
                        vflag = 1
        break