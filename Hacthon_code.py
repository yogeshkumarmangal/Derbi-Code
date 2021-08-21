# Reading Iamge
# Show Iamge
# Size and Shape Of Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image 
list1=[]
list2=[]
list3=[]
# import Library
for i in range(1,2501):
    #path1=str("D:\Derbhi Hackthon\TB_Chest_Radiography_Database\Tuberculosis")
    print(i)
    path1=str("D:\Derbhi Hackthon\PE")
    path2=str("\PE-")
    path3=str(i)
    path4=str(".jfif")
    Imagename=path1+path2+path3+path4
    img = cv2.imread(Imagename,cv2.IMREAD_COLOR)
    h, w, c = img.shape
    y=int(h*16/100)
    x=int(w*16/100)
    crop_image = img[x:w,y:h]
    ##cv2.imshow('Image',img)
    grayimg = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    h, w, c = img.shape
    print("Height of Image = ",h)
    print('Width Of Image =  ',w)
    print('Channel of Image = ',c)


    #Apply Gaussian Filter
    Gaussian_Iamge = cv2.GaussianBlur(grayimg,(5,5),0)
    gh, gw= Gaussian_Iamge.shape
    print("\nHeight of Gaussian Iamge = ",gh)
    print('Width Of Gaussian Image =  ',gw)
    fig = plt.figure(figsize=(10,7))
    rows = 1
    columns = 2
    fig.suptitle("Tuberculosis", fontsize="x-large")
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original Iamge")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
      
    # showing image
    plt.imshow(Gaussian_Iamge)
    plt.axis('off')
    plt.title('Gaussian Smoothing')
    plt.savefig('Tuberculosis.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    ## Canny Edge Detection
    edges = cv2.Canny(Gaussian_Iamge,30,30)
    plt.imshow(edges)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    df = pd.DataFrame(edges/255)
    Descibe_Data=df.describe()
    Descibe_Data.to_excel('TB-ROI-Describe-Data.xlsx')
    df1=Descibe_Data.mean(axis=1)
    med1=df1.iloc[1:len(df1)].mean(axis=0)
    df2=Descibe_Data.median(axis=1)
    med2=df2.iloc[1:len(df2)].mean(axis=0)
    df3=Descibe_Data.std(axis=1)
    med3=df3.iloc[1:len(df3)].mean(axis=0)
    list1.append(med1)
    list2.append(med2)
    list3.append(med3)
list5=sum(list1)/len(list1)
list6=sum(list2)/len(list2)
list7=sum(list3)/len(list3)
list4=['Mean','Median','std']
list8=[round(list5,4),round(list6,4),round(list7,4)]
res = dict(zip(list4, list8))
new = pd.DataFrame.from_dict(res,orient ='index')
new=pd.DataFrame({'Mean' : [list8[0]],
                                'Median' : [list8[1]],
                                'Standred Deviation' : [list8[2]] }, 
                                columns=['Mean','Median', 'Standred Deviation'])

new.to_excel('PE.xlsx',index=None)
