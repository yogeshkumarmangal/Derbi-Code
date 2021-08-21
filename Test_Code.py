# Reading Iamge
# Show Iamge
# Size and Shape Of Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image 
# import Library
for i in range(510,521):
    path1=str("D:\Derbhi Hackthon\TB_Chest_Radiography_Database\Tuberculosis")
    print(i)
    #path1=str("D:\Derbhi Hackthon\PE")
    path2=str("\Tuberculosis-")
    path3=str(i)
    path4=str(".png")
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
    df4=pd.read_excel('TBdata.xlsx')
    df5=pd.read_excel('coviddata.xlsx')
    df6=pd.read_excel('PE.xlsx')
    data_point_TB=np.array([df4['Mean'],df4['Median']])
    data_point_Covid=np.array([df5['Mean'],df5['Median']])
    data_point_PE=np.array([df6['Mean'],df6['Median']])
    daat_point_test_image=np.array([round(med1,4),round(med2,4)])
    Euclidean_distance_TB = round(np.linalg.norm(data_point_TB - daat_point_test_image),4)
    Euclidean_distance_Covid = round(np.linalg.norm(data_point_Covid - daat_point_test_image),4)
    Euclidean_distance_PE = round(np.linalg.norm(data_point_PE - daat_point_test_image),4)
    list9=[str("TB"),str("Covid"),str("PE")]
    list8=[Euclidean_distance_TB, Euclidean_distance_Covid,Euclidean_distance_PE]
    new_data=pd.DataFrame({'TB' : [list8[0]],
                               'Covid' : [list8[1]],
                                     "PE" : [list8[2]]}, 
                              columns=['TB', 'Covid','PE'])
    
    df7=pd.read_excel('testdatas.xlsx')
    df8=df7.append(new_data)
    df8.to_excel('testdatas.xlsx',index=None)
##    mean1=abs(med1-df4['Mean'])
##    mean2=abs(med2-df4['Median'])
##
##    mean3=abs(med1-df5['Mean'])
##    mean4=abs(med2-df5['Median'])
##
##    mean6=abs(med1-df6['Mean'])
##    mean7=abs(med2-df6['Median'])


##list6=sum(list2)/len(list2)
##list7=sum(list3)/len(list3)
##list4=['Mean','Median','std']
##list8=[round(list5,4),round(list6,4),round(list7,4)]
##res = dict(zip(list4, list8))
##new = pd.DataFrame.from_dict(res,orient ='index')
##new=pd.DataFrame({'Mean' : [list8[0]],
##                                'Median' : [list8[1]],
##                                'Standred Deviation' : [list8[2]] }, 
##                                columns=['Mean','Median', 'Standred Deviation'])
##
##
