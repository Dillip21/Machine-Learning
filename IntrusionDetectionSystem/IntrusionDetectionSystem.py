#234102304- DILLIP S-IntrusionDetectionSystem

import cv2
import numpy as np
import os


def detectChanges(img1, img2, eta_c=16):

    diff = cv2.absdiff(img1, img2)    
    chebyshev_distance = np.max(diff, axis=2)    
    change_mask = chebyshev_distance > eta_c
    withShadow = change_mask.astype(np.uint8) * 255
    noShadow = removeShadows(withShadow)
    return noShadow

def removeShadows(withShadow):
    # Apply morphological operations to the binary image
    kernel = np.ones((6, 6), np.uint8)
    morphed_image = cv2.morphologyEx(withShadow, cv2.MORPH_OPEN, kernel)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed_image, connectivity=8)
    
    # Remove shadow regions based on area or intensity threshold
    for label in range(1, num_labels):  # Skip background label 0
        region_area = stats[label, cv2.CC_STAT_AREA]
        # Example: Remove regions with area less than 150 pixels
        if region_area < 150:
            morphed_image[labels == label] = 0
    
    return morphed_image

def noiseRemoval(change_mask, neighborhood_size=(5, 5), eta_v=0.6):

    kernel = np.ones(neighborhood_size, np.uint8)    
    dilated_mask = cv2.morphologyEx(change_mask.astype(np.uint8),cv2.MORPH_OPEN, kernel)    
    voting_result = dilated_mask / np.sum(kernel)
    voting_result = voting_result > eta_v
    
    return voting_result.astype(np.uint8) * 255


if __name__ == "__main__":
    iref=cv2.imread(r"Images/AirstripRunAroundFeb2006_1300.bmp")
    #cv2.imshow("iref",iref)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    x=1300
    while True:
        img=cv2.imread(rf"Images\AirstripRunAroundFeb2006_{x}.bmp")
        if img is None:
            cv2.destroyAllWindows()
            break
        x+=1
        changeDetectedImage=detectChanges(iref,img)
        output=noiseRemoval(changeDetectedImage)
        #print(output)
        cv2.imshow("Input",img)
        cv2.imshow('Final Output', output)                                                                                                                                                                                                                                                                      
        cv2.waitKey(10)