"""
Implement model blur classification for product
"""

import pickle
import numpy as np
from skimage import feature

def lbph(image, numPoints, eps = 1e-7):
    
    """
    Implement LBPH algorithm 
    Parameters:
        - image: gray image 
        - numPoints: numPoints of LBPH
    Return:
        - histogram of image
    """
    lbp = feature.local_binary_pattern(image, 24, 8, method="uniform")
    
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
	# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
	# return the histogram of Local Binary Patterns
    return hist

class Blurtor:

    def __init__(self, model_path):

        self.clf = pickle.load(open(model_path, 'rb'))
    
    def predict(self, image):

        """
        Predict a image has blur or no 
        Required: image is a gray image 
        """
        try:
            
            hist =  lbph(image, 14)
            pred = self.clf.predict([hist])
            if pred[0] == '0':
                return True 
            return True

        except:
            return None