#DILLIP_S - 234102304 - EE722

import cv2
import numpy as np

def initBackgroundModel(img, initVar):
    meanBakImg = img.astype(np.float32)
    varBakImg = np.full_like(img, initVar, dtype=np.float32)
    #print(meanBakImg.shape, varBakImg.shape)
    return meanBakImg, varBakImg

def foregroundExtraction(img, meanBakImg, varBakImg, _lambda):
    noisyChangeImg = np.abs(img - meanBakImg)/np.sqrt(varBakImg)
    mask = np.max(noisyChangeImg, axis=2) > _lambda
    return mask.astype(np.uint8) * 255

def updateBackgroundModel(meanBakImg, varBakImg, img, alpha):
    meanBakImg = (1-alpha) * meanBakImg + alpha*img
    varBakImg =(1- alpha) * (varBakImg+ alpha*(img-meanBakImg)**2)
    return meanBakImg, varBakImg

def detectShadow(changeImg, modThreshold):
    # Apply morphological operations to the binary image
    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.dilate(changeImg, kernel)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed_image, connectivity=8)
    
    # Remove shadow regions based on area or intensity threshold
    for label in range(1, num_labels):  # Skip background label 0
        region_area = stats[label, cv2.CC_STAT_AREA]
        # Example: Remove regions with area less than threshold
        if region_area < modThreshold:
            morphed_image[labels == label] = 0
    
    return morphed_image    # Shadow detection implementation


def noiseRemoval(change_mask, neighborhood_size=(4, 4), eta_v=0.6):

    kernel = np.ones(neighborhood_size, np.uint8)    
    dilated_mask = cv2.erode(change_mask.astype(np.uint8), kernel)    
    voting_result = dilated_mask / np.sum(kernel)
    voting_result = voting_result > eta_v
    
    return voting_result.astype(np.uint8) * 255



if __name__ == '__main__':

    # Initialization
    
    initVar = 16  # Initial variance value
    _lambda = 3  # Lambda threshold for foreground extraction
    alpha = 0.00001 # Learning rate for background model update

    
    iref=cv2.imread(r"Images/AirstripRunAroundFeb2006_1300.bmp")
    meanBakImg, varBakImg = initBackgroundModel(iref, initVar)
    
    x=1300
    while True:
        img=cv2.imread(rf"Images\AirstripRunAroundFeb2006_{x}.bmp")
        if img is None:
            cv2.destroyAllWindows()
            break
        x+=1
        
        
        # Operation
        noisyChangeImg = foregroundExtraction(img, meanBakImg, varBakImg, _lambda)
        changeImg = noiseRemoval(noisyChangeImg)  # Noisy change image can be passed to voting function if needed
        #changeImgout = changeImg.round().astype(np.uint8) * 255
        shadowDetChangeImg = detectShadow(changeImg, 150)  # Modulation threshold for shadow detection
        meanBakImg, varBakImg = updateBackgroundModel(meanBakImg, varBakImg, img, alpha)
        meanImg = meanBakImg.round().astype(np.uint8)

        # Display the processed frame  
        
        cv2.imshow('Frame', img)
        cv2.imshow('Mean Image', meanImg)
        cv2.imshow('Foreground Mask', noisyChangeImg )  # Scaling to 0-255 for visualization
        cv2.imshow('Intrusion',shadowDetChangeImg)


        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close OpenCV windows
    cv2.destroyAllWindows()







    

#DILLIP_S - 234102304 - EE722