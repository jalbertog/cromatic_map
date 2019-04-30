import cv2
import numpy as np
import sys

def normalize(img,real,value=74):
    h,w,d = img.shape
    norm = real - value
    for i in xrange(h):
        for j in xrange(w):
            img[i,j,0] = int(img[i,j,0]+norm)
    return img

c = len(sys.argv)
im_in = 'diente.jpg'
im_temp = 'grey.jpg'
SAMPLE_WINDOW = 64
if c >= 2:
    im_in = sys.argv[1]
if c >=3:
    im_temp = sys.argv[2]

source = cv2.imread(im_in)
grey_card = cv2.imread(im_temp)
h,w,d = grey_card.shape
h_sample = int(h/2 - SAMPLE_WINDOW)
w_sample = int(w/2 - SAMPLE_WINDOW)
sample_grey = grey_card[h_sample:(h_sample+2*SAMPLE_WINDOW),w_sample:(w_sample+2*SAMPLE_WINDOW)]
cv2.imwrite('sample.jpg',sample_grey)
source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
sample_grey = cv2.cvtColor(sample_grey, cv2.COLOR_BGR2LAB)
L_value = np.mean(sample_grey[:,:,0:1])
print (L_value)
source = normalize(source,L_value,188)



matched = cv2.cvtColor(source, cv2.COLOR_LAB2BGR)

im_out = 'LAB_out_'+im_in
if c >= 4:
    im_out = sys.argv[3]
cv2.imwrite(im_out,matched)
