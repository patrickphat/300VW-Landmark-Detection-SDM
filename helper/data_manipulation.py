import cv2
import numpy as np
from skimage.feature import hog

def load_and_preprocess(img_path):
    # Load image
    img = cv2.imread(img_path,0)

    # Resize image
    img = cv2.resize(img,(320,180))

    # Extract hog feature
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=False)

    # Expand first dimension
    feature = np.expand_dims(feature,0)

    return img,feature

def draw_landmark(image,y_hat):

    y_hat = y_hat[0].reshape(68,2)

    for annot in y_hat:
        annot = tuple(annot.astype('uint16'))
        image = cv2.circle(image,annot,1,255,-1) 

    return image 
