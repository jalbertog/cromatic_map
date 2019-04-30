import cv2
import numpy as np

class GrayWorldBalance:

    def white_balance(self,img,debug=False):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        if debug:
            cv2.imwrite('balance_grey_world.JPG',result)
        return result

    def apply(self,img,debug=False):
        return self.white_balance(img,debug)

if __name__ == "__main__":   
    import sys  
    c = len(sys.argv)
    im_in = 'diente.jpg'
    if c >= 2:
        im_in = sys.argv[1]

    image = cv2.imread(im_in)
    whtB = GrayWorldBalance()
    wht = whtB.apply(image,True)
