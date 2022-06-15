#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage import color
from tqdm import tqdm
import copy
import sys
#%%
def energy_function1(image):
    newImage = np.copy(image.astype(np.uint8))
    grayImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    #finding the gradient in x and y dir
    sobelx = cv2.Sobel(grayImage,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(grayImage,cv2.CV_64F,0,1,ksize=5)
    return abs(sobelx) + abs(sobely)

def energy_function2(image):
        newImage = np.copy(image.astype(np.float64))
        b, g, r = cv2.split(newImage)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy

def get_cumulative_energy_function(energy_map):
    new_energy_map = np.copy(energy_map)
    rows,columns = energy_map.shape
    for r in range(1,rows):
        for c in range(columns):
            new_energy_map[r,c] = \
            new_energy_map[r,c]+np.amin(new_energy_map[r-1,max(c-1,0):min(c+2,columns)])
    return new_energy_map

def my_argmin(array,start_index,end_index,limit):
    min = np.max(array)+15
    argmin = -1

    if(end_index > limit):
        end_index = limit

    for i in range(start_index,end_index,1):
        if(array[i] < min):
            min = array[i]
            argmin = i
    if(argmin > limit):
        print("How come Illegal? ",argmin,limit,array[start_index:end_index])
    return argmin

def get_seam(cumulative_energy_map):
    rows,columns = cumulative_energy_map.shape

    #From first row till last,has column values
    seam_path = np.zeros((rows,), dtype=np.uint32)
    seam_path[-1] = np.argmin(cumulative_energy_map[-1])

    #BackTracking
    for r in range(rows-2,-1,-1):
        previous_col = seam_path[r+1]
        #if previous_col==984:
            #print(previous_col)
        if previous_col == 0:
            seam_path[r] = my_argmin(cumulative_energy_map[r],0,2,columns)
            #seam_path[r] = np.argmin(cumulative_energy_map[r,0:2])
        else:
            #if previous_col==984:
                #print("Yes",np.argmin(cumulative_energy_map[r,previous_col-1:min(previous_col+2,columns)]))
            seam_path[r] = my_argmin(cumulative_energy_map[r],previous_col-1,previous_col+2,columns)
            #seam_path[r] = np.argmin(cumulative_energy_map[r,previous_col-1:min(previous_col+2,columns)])
    #print("Within Seam",columns,np.min(seam_path),np.max(seam_path))
    return seam_path

def delete_seam(newImage,seam_path):
    rows,columns = newImage.shape[0:2]

    #One column id getting deleted
    updatedImage = np.zeros((rows, columns - 1, 3))

    for r in range(rows):
        c = seam_path[r]
        updatedImage[r, :, 0] = np.delete(newImage[r, :, 0], [c])
        updatedImage[r, :, 1] = np.delete(newImage[r, :, 1], [c])
        updatedImage[r, :, 2] = np.delete(newImage[r, :, 2], [c])
    return updatedImage

def add_seam(newImage, seam_path):
        rows, columns = newImage.shape[0:2]
        updatedImage = np.zeros((rows, columns + 1, 3),dtype=np.uint8)
        for r in range(rows):
            c = seam_path[r]
            for channel in range(3):
                if c == 0:
                    p = np.average(newImage[r, c: c + 2, channel])
                    if(p>255.0):
                        p = 255.0
                    p = p.astype(np.uint8)

                    updatedImage[r, c, channel] = p
                    updatedImage[r, c + 1:, channel] = newImage[r, c:, channel]
                else:
                    p = np.average(newImage[r, c - 1: c + 2, channel])
                    if(p>255.0):
                        p = 255.0
                    p = p.astype(np.uint8)
                    updatedImage[r, : c, channel] = newImage[r, :c, channel]
                    updatedImage[r, c, channel] = p
                    updatedImage[r, c + 1:, channel] = newImage[r, c:, channel]
        return updatedImage

def update_seam_list(all_seams_list, added_seam_path,cond=1):
        updated_seam_list = []
        #print("Len SeamList :",len(all_seams_list))
        for seam_path in all_seams_list:
            if cond == 1:
                new_seam_path = copy.deepcopy(seam_path)
                new_seam_path[np.where(seam_path >= added_seam_path)] += 2
            elif cond ==2:
                #print("Before Min : ",np.min(seam_path)," Max : ",np.max(seam_path))
                new_seam_path = copy.deepcopy(seam_path)
                new_seam_path[np.where(seam_path >= added_seam_path)] += 1
                #print("After Min : ",np.min(seam_path)," Max : ",np.max(seam_path))
            updated_seam_list.append(new_seam_path)
        return updated_seam_list

#Lets increase by column
def enlarge_column(image,num_of_cols):
    newImage = np.copy(image)
    #print("1",newImage.dtype)
    secondImage = np.copy(image)
    #print("2",secondImage.dtype)
    all_seams_list = []
    global temp_seam_list
    print("Fetch all Seams")
    for c in tqdm(range(num_of_cols)):
        energy_map = energy_function1(newImage)
        cumulative_energy_map = get_cumulative_energy_function(energy_map)
        seam_path = get_seam(cumulative_energy_map)
        all_seams_list.append(seam_path)
        #To get next non-overlapping seam
        newImage = delete_seam(newImage,seam_path)
        #print("Delete Image Shape:",newImage.shape[0:2])

    temp_seam_list = copy.deepcopy(all_seams_list)

    print("Add all Seams")
    for c in tqdm(range(num_of_cols)):
        seam_path = all_seams_list.pop(0)
        secondImage = add_seam(secondImage,seam_path)
        all_seams_list = update_seam_list(all_seams_list,seam_path)
        #print("Add Image Shape:",secondImage.shape[0:2])

    return secondImage,temp_seam_list

def plotseam(img,seam_list):
  img3 = np.copy(img)
  #global new_seam_list
  new_seam_list = seam_list.copy()

  for c in range(len(new_seam_list)):
      current_seam = new_seam_list.pop(0)
      for i in range(img.shape[0]):
        img3[i,current_seam[i]] = [0,0,255]

      new_seam_list = update_seam_list(new_seam_list,current_seam,cond=2)

  plt.figure(figsize=(10,10))
  plt.title('Seam')
  plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))


def rotate_image( image, ccw):
    m, n, ch = image.shape
    output = np.zeros((n, m, ch),dtype=np.uint8)
    if ccw:
        image_flip = np.fliplr(image)
        for c in range(ch):
            for row in range(m):
                output[:, row, c] = image_flip[row, :, c]
    else:
        for c in range(ch):
            for row in range(m):
                output[:, m - 1 - row, c] = image[row, :, c]
    return output

#%%
# generate Gaussian pyramid
import math

def checkIfEvenDimensions(img):
    row,column = img.shape[:2]
    return (row%2),(column%2)

def getNumSeamsPerLevel(numLevels,numSeams):
    sigma = 0
    numSeamsPerLevel = []
    for i in range(numLevels-1,-1,-1):
        nI = math.floor((numSeams -  sigma) / math.pow(2.0,i))
        sigma += nI * math.pow(2,i)
        numSeamsPerLevel.append(nI)
        #print("Level : ",i,numSeamsPerLevel)
    numSeamsPerLevel.reverse()
    return numSeamsPerLevel

def getGaussianPyramids(img,numLevels):
    newImage = img.copy()
    level = 0
    removedRows = { }
    removedCols = { }

    isNotRowEven,isNotColEven = checkIfEvenDimensions(newImage)
    isEvenRows = [isNotRowEven]
    isEvenColumns = [isNotColEven]

    #Remove First row
    if isNotRowEven:
        removedRows[level] = newImage[0,:,:]
        newImage = newImage[1:,:,:]

    #Remove First Column
    if isNotColEven:
        removedCols[level] = newImage[:,0,:]
        newImage = newImage[:,1:,:]

    gaussianPyramids = [newImage]

    for i in range(numLevels-1):
        level += 1
        newImage = cv2.pyrDown(newImage)
        #ignore Last level Check
        if level == numLevels-1 :
            isEvenRows.append(0)
            isEvenColumns.append(0)
        else:
            isNotRowEven,isNotColEven = checkIfEvenDimensions(newImage)

            isEvenRows.append(isNotRowEven)
            isEvenColumns.append(isNotColEven)


            if isNotRowEven:
                removedRows[level] = newImage[0,:,:]
                newImage = newImage[1:,:,:]

            if isNotColEven:
                removedCols[level] = newImage[:,0,:]
                newImage = newImage[:,1:,:]

        gaussianPyramids.append(newImage)
    return  gaussianPyramids,isEvenRows,removedRows,isEvenColumns,removedCols
#%%
# Display Gaussian pyramid
def plotPyramids(gaussianPyramids,numLevels):
    fig=plt.figure(figsize=(50, 50))
    columns = 1
    rows = numLevels
    for i in range(1, columns*rows +1):
        img = gaussianPyramids[i-1]
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i)
        plt.title("Level ",i)
        plt.imshow(img)
    plt.show()
#%%
def getNextLevelSeamPath(seam_path):
    new_seam_path = []
    for i in range(len(seam_path)):
        new_seam_path.append(seam_path[i]*2)
        new_seam_path.append(seam_path[i]*2)
    return np.array(new_seam_path,dtype='uint32')

#%%

def enlargeColumnsFast(image,numLevels,numberOfSeams):
    img = copy.deepcopy(image)
    gaussianPyramids,isEvenRows,removedRows,isEvenColumns,removedCols = getGaussianPyramids(img,numLevels)
    len(gaussianPyramids)
    numSeamsPerLevel = getNumSeamsPerLevel(numLevels,numberOfSeams)

    #%%
    #numLevels = 3
    all_seams_list = []
    for i in range(numLevels-1,-1,-1):
        myImg = gaussianPyramids[i]

        newImage = np.copy(myImg)

        if isEvenColumns[i]:
            rows, columns = newImage.shape[0:2]
            updatedImage = np.zeros((rows, columns + 1, 3), dtype=np.uint8)
            for channel in range(3):
                updatedImage[:, 0, channel] = (removedCols[i])[:, channel]
                updatedImage[:, 1:, channel] = newImage[:, :, channel]
            newImage = updatedImage

        if isEvenRows[i]:
            rows, columns = newImage.shape[0:2]
            updatedImage = np.zeros((rows + 1, columns, 3), dtype=np.uint8)
            for channel in range(3):
                updatedImage[0, :, channel] = (removedRows[i])[:, channel]
                updatedImage[1:, :, channel] = newImage[:, :, channel]
            newImage = updatedImage


        tmpImg = np.copy(newImage)




        new_seams_list = []
        for k in range(len(all_seams_list)):
            seam_path = all_seams_list.pop(0)



            seam_path_for_next_level_1 =getNextLevelSeamPath(seam_path)

            if isEvenColumns[i]:
                new_seam_path_next_level = [element + 1 for element in seam_path_for_next_level_1]
                seam_path_for_next_level_1 = np.array(new_seam_path_next_level,dtype='uint32')

            if isEvenRows[i]:
                seam_path_new = [seam_path_for_next_level_1[0]]
                seam_path_new.extend(seam_path_for_next_level_1.tolist())
                seam_path_for_next_level_1 = np.array(seam_path_new,dtype='uint32')


            seam_path_for_next_level_2 = copy.deepcopy(seam_path_for_next_level_1)
            newImage = delete_seam(newImage,seam_path_for_next_level_1)
            newImage = delete_seam(newImage,seam_path_for_next_level_2)


            new_seams_list.append(seam_path_for_next_level_1)
            #updated_seam_path_for_next_level_2 = [element + 1 for element in seam_path_for_next_level_2]
            updated_seam_path_for_next_level_2 = seam_path_for_next_level_1
            new_seams_list.append(updated_seam_path_for_next_level_2)
            #all_seams_list = update_seam_list(all_seams_list,seam_path,cond=1)

        all_seams_list = copy.deepcopy(new_seams_list)



        for j in range(numSeamsPerLevel[i]):
            energy_map = energy_function1(newImage)
            cumulative_energy_map = get_cumulative_energy_function(energy_map)
            seam_path = get_seam(cumulative_energy_map)
            all_seams_list.append(seam_path)
            newImage = delete_seam(newImage,seam_path)
        
        img1 = np.copy(newImage)

        if i==0:
            #plotseam(tmpImg, all_seams_list)
            seamCount = len(all_seams_list)
            assert seamCount == numberOfSeams
            for c in tqdm(range(seamCount)):
                seam_path = all_seams_list.pop(0)
                tmpImg = add_seam(tmpImg,seam_path)
                all_seams_list = copy.deepcopy(update_seam_list(all_seams_list,seam_path))

            secondImage = copy.deepcopy(tmpImg)
            return secondImage,img1

#%%
def noraml_enlarge(myImage, colSize, rowSize):
    img = copy.deepcopy(myImage)
    print("Normal Enlarging Column...")
    fastEnlargedImageCol,_ = enlarge_column(img, colSize)
    rowsE, columnsE = fastEnlargedImageCol.shape[0:2]
    print("Rows : ", rowsE, "Columns : ", columnsE)

    print("Normal Enlarging Row...")
    rImage = rotate_image(fastEnlargedImageCol, 0)
    fastEnlargedImageRow,_ = enlarge_column(rImage, rowSize)

    OrigImage = rotate_image(fastEnlargedImageRow, 1)
    rowsO, columnsO = OrigImage.shape[0:2]
    print("Rows : ", rowsO, "Columns : ", columnsO)
    return OrigImage

#%%
def fast_enlarge(myImage,colLevels,colSize,rowLevels,rowSize):
    img = copy.deepcopy(myImage)
    print("Fast reducing Column...")
    _,img2 = enlargeColumnsFast(img,colLevels,colSize)
    #rowsE,columnsE = fastEnlargedImageCol.shape[0:2]
    #print("Rows : ",rowsE,"Columns : ",columnsE)

    print("Fast reducing Row...")
    rImage = rotate_image(img2,0)
    _,img3 = enlargeColumnsFast(rImage,rowLevels,rowSize)


    OrigImage = rotate_image(img3,1)
    rowsO,columnsO = OrigImage.shape[0:2]
    print("Rows : ",rowsO,"Columns : ",columnsO)
    return OrigImage

#%%
img = cv2.imread('/content/drive/MyDrive/COL783/1024px-Broadway_tower_edit.jpg')
#normalImage = noraml_enlarge(img,10,10)
colsize = np.int(sys.argv[1])
rowsize = np.int(sys.argv[2]) 
fastImage = fast_enlarge(img,5,colsize,5,rowsize)

#cv2.imwrite('/Users/akshay/Documents/GuassinaPyramid/data/normal_enlarge_output.png',normalImage)
cv2.imwrite('/content/drive/MyDrive/COL783/fastseamresult.jpg',fastImage)



#%%


