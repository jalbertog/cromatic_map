import sys
import cv2
from skimage import color, io, img_as_ubyte,img_as_float
from graySkimage import BalanceIlumination
from whitebalgreyworld import GrayWorldBalance
from extractor import Segmentation, MeanStimator
from morphologicalThr import MorphologicalSegmentation

class Preprocesing:
    
    def __init__(self,cols=1,rows=3):
        self.cols = cols
        self.rows = rows

    def skimage2opencv(self,img,ubyte=True):
        res = img[:, :, ::-1]
        return img_as_ubyte(res) if ubyte else img_as_float(res)

    def process(self,img_in, template,true_color=True,name_out='split.jpg'):
        source = io.imread(img_in)
        grey_card = io.imread(template)

        #bIlum = BalanceIlumination()
        #res = bIlum.apply(source,grey_card)
        bIlum = GrayWorldBalance()
        image = bIlum.apply(self.skimage2opencv(source),True)
        res = self.skimage2opencv(image,False)
        io.imsave('save.png',res)
        #image = self.skimage2opencv(res)
        #sm = Segmentation()
        #mask, min_x, min_y, max_x, max_y = sm.apply(image)
        sm = MorphologicalSegmentation()
        mask, min_x, min_y, max_x, max_y = sm.apply(image)
        cv2.imwrite('mask.png',mask)
        square = [min_x, min_y, max_x, max_y]
        print('square = {}'.format(square))       
    
        mSt = MeanStimator(true_color)
        values = mSt.stimate(res,mask,square,self.cols,self.rows,name_out)
        print(values)

        return values;
        
if __name__ == "__main__":

    c = len(sys.argv)
    im_in = 'diente1.jpg'
    im_temp = 'grey.jpg'
    name_out = 'result.jpg'
    true_color = True
    cc = 3
    rr = 1
    if c >= 2:
        im_in = sys.argv[1]
    if c >=3:
        im_temp = sys.argv[2]
    if c >=4:
        cc = int(sys.argv[3])
    if c >=5:
        rr = int(sys.argv[4])
    if c >=6:
        true_color = sys.argv[5] == 'true'

    if(c >= 7):
        name_out = sys.argv[6]

    pp = Preprocesing(cc,rr)
    pp.process(im_in,im_temp,true_color,name_out)

    print('finish')
