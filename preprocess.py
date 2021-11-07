import cv2
import os
def selfie_preprocess(path):
    # crop -> grayscale -> resize to 32x32
    for i, im in enumerate(os.listdir(path)):
        im_path = os.path.join(path, im)
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32,32))
        cv2.imwrite(os.path.join(path, f'{i}.jpg'), img)


    
if __name__=='__main__':
    selfie_preprocess('./selfie')