import cv2
import numpy as np
import sys
from skimage import color, io, img_as_ubyte


class BalanceIlumination:
    def __init__(self, sample_window = 10):
        self.SAMPLE_WINDOW = sample_window
    
    def normalize(self,img,real,value=74):
        h,w,d = img.shape
        norm = real - value
        #for i in xrange(h):
         #   for j in xrange(w):
        #        img[i,j,0] = int(img[i,j,0]+norm)
        img[:,:,0] = img[:,:,0]+norm
        return img

    def apply(self, source, grey_card, grey_card_l_value = 74,debug=True):
        h,w,d = grey_card.shape
        h_sample = int(h/2 - self.SAMPLE_WINDOW)
        w_sample = int(w/2 - self.SAMPLE_WINDOW)
        sample_grey = grey_card[h_sample:(h_sample+2*self.SAMPLE_WINDOW),w_sample:(w_sample+2*self.SAMPLE_WINDOW)]
        source = color.rgb2lab(source)
        sample_grey = color.rgb2lab(sample_grey)
        L_value = np.mean(sample_grey[:,:,0:1])
        source = self.normalize(source,L_value,grey_card_l_value)

        matched = color.lab2rgb(source)
        if debug:
            print('finish Balance')
            im_out = 'Img_balance.jpg'
            io.imsave(im_out,matched)

        return matched

if __name__ == "__main__":

    c = len(sys.argv)
    im_in = 'diente1.jpg'
    im_temp = 'grey.jpg'
    SAMPLE_WINDOW = 64
    if c >= 2:
        im_in = sys.argv[1]
    if c >=3:
        im_temp = sys.argv[2]

    source = io.imread(im_in)
    grey_card = io.imread(im_temp)

    bIlum = BalanceIlumination()
    res = bIlum.apply(source,grey_card)

