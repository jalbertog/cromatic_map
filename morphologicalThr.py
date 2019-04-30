import cv2
import numpy as np
from skimage import morphology, filters

def open_by_reconstruction(src, iterations = 1, ksize = 3):
    # first erode the source image
    eroded = cv2.erode(src, np.ones((ksize,ksize), np.uint8), iterations=iterations)

    # Now we are going to iteratively regrow the eroded mask.
    # The key difference between just a simple opening is that we
    # mask the regrown everytime with the original src.
    # Thus, the dilated mask never extends beyond where it does in the original.
    this_iteration = eroded
    last_iteration = eroded
    while (True):
        this_iteration = cv2.dilate(last_iteration, np.ones((ksize,ksize), np.uint8), iterations = 1)
        this_iteration = this_iteration & src
        if np.array_equal(last_iteration, this_iteration):
            # convergence!
            break
        last_iteration = this_iteration.copy()

    return this_iteration

class MorphologicalSegmentation:
            
    def apply(self,color_im,method = 'r_small_objects', remove_spots = True, level = 200, debug=True):
        return self.optimalThreshold(color_im,method, remove_spots, level, debug)

    def optimalThreshold(self,color_im,method = 'r_small_objects', remove_spots = True, level = 200, debug=True):
        im = cv2.cvtColor(color_im,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if debug:
            cv2.imwrite('binary.jpg',thresh)

        thresh = filters.median(thresh,morphology.square(7))
        cv2.imwrite('binary_med.jpg',thresh)
        #MORPHOLOGICAL_REMOVE_SMALL_OBJECTS
        imglab = morphology.label(thresh) # create labels in segmented image
        min_s = (int)(im.shape[0])
        try:
            cleaned = morphology.remove_small_objects(imglab, min_size=min_s, connectivity=8)
        except UserWarning:
            pass
        res_small = np.zeros((cleaned.shape)) # create array of size cleaned
        res_small[cleaned > 0] = 255 
        res_small= np.uint8(res_small)
        if debug:
            cv2.imwrite("cleaned.jpg", res_small)
 
        #MORPOLOGICAL_OPENING_OPENCV
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 32))
        opened_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        if debug:
            cv2.imwrite("opened_mask.jpg", opened_mask)

        if remove_spots:
            blurred = cv2.GaussianBlur(im, (11, 11), 0)
            spots = cv2.threshold(blurred, level, 255, cv2.THRESH_BINARY)[1]
            res_small = res_small - spots
            opened_mask = opened_mask - spots
            if debug:
                cv2.imwrite("cleaned_spots.jpg", res_small)
                cv2.imwrite("opened_mask_spots.jpg", opened_mask)
        img, contours, h = cv2.findContours(res_small,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        max_c = []
        max_a = 0
        result = np.zeros(img.shape)
        for contour in contours:
            if cv2.contourArea(contour) > max_a:
                max_c = contour
                max_a = cv2.contourArea(contour)

        bouding = cv2.boundingRect(max_c)
        print(bouding)
        if debug:
            masked_img = cv2.bitwise_and(color_im, color_im, mask=opened_mask)
            masked_img2 = cv2.bitwise_and(color_im, color_im, mask=res_small)
            cv2.imwrite("masked_img.jpg", masked_img)
            cv2.imwrite("masked_img_small.jpg", masked_img2)
        min_x = bouding[1]
        min_y = bouding[0]
        w_x = bouding[1]+ bouding[3]
        h_y = bouding[0]+ bouding[2]
        return res_small,min_x,min_y,w_x,h_y
if __name__ == "__main__":  
    import sys
   
    c = len(sys.argv)
    im_in = 'Diente_d_3.JPG'
    if c >= 2:
        im_in = sys.argv[1]

    image = cv2.imread(im_in)
    sm = MorphologicalSegmentation()
    mask, min_x, min_y, max_x, max_y = sm.apply(image)

