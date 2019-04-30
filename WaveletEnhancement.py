import numpy as np

import pywt
import cv2


class WaveletEnhancement(object):

    def __init__(self,m_ini=0,M_ini=255,automatic_mM = True):
        self.m = m_ini
        self.M = M_ini
        self.automatic_mM = automatic_mM

    def enhanceSaturation(self, image):
        return cv2.equalizeHist(image)
    
    def selectnM(self,cA):
        if self.automatic_mM == False:
            return

        self.m = 30
        self.M = 200
        
        [m1,M1] = np.quantile(cA,[0.05,0.95])
        self.m = np.min([self.m,m1])
        self.M = np.max([self.M,M1])
        

        
    def applyWaveTransformEnhancement(self,original):
        cA, (LH, HL, HH) = pywt.dwt2(original, 'sym4')
        self.selectnM(cA)

        K1 = self.m-1
        K2 = self.M+1
        cv2.imwrite('salida.jpg',cA)
        values = np.where((cA > self.m) & (cA < self.M))
        print(self.M-self.m)
        print (cA[values].shape)
        cA[values] = np.log(((cA[values]-K1)/(K2-cA[values])))
        R_min = np.min(cA[values])
        R_max = np.max(cA[values])
        cA[values] = ((cA[values]-R_min)/(R_max-R_min))
        print (cA[values])
        cA[values] = cA[values]*(self.M-self.m) + self.m
        
        print (cA[values])

        res = pywt.idwt2((cA, (LH, HL, HH)),'sym4')
        return np.uint8(res)

    def apply(self, original):
        hsv = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        (sx,sy,_) = hsv.shape
        hsv = cv2.resize(hsv,(512,512))
        H,S,V = cv2.split(hsv)
        print (np.max(V))
        S = self.enhanceSaturation(S)
        V = self.applyWaveTransformEnhancement(V)
        result = cv2.merge([H,S,V])
        result = cv2.resize(hsv,(sy,sx))
        result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
        cv2.imwrite('resultado.jpg',result)


if __name__ == "__main__":  
    import sys
   
    c = len(sys.argv)
    im_in = 'imagen2.png'
    if c >= 2:
        im_in = sys.argv[1]

    image = cv2.imread(im_in)
    wvlt = WaveletEnhancement()
    wvlt.apply(image)
