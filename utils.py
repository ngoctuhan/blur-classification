import os, cv2, numpy as np
from numpy.lib.histograms import histogram 
from skimage import feature
def historgram(image, binn, eps = 1e-7):

    """
    Calculator histogram of gray image 
    Paramters:
        - image: input image with shape = (width, height)

    Return:
        - Output: a vector 256 elements 
    """
    (hist, _) = np.histogram(image,bins=np.arange(0, binn))
	# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
    
    return hist

def historgram_RGB(image, binn):

    
    """
    Calculator historgram with RGB image 
    """
    hist = []
    for chanel in range(3):
        element = image[:,:,chanel]
        hist.append(historgram(element, binn))
    
    result = np.array(hist).reshape(-1,)
    return result

def lbph(image, numPoints, eps = 1e-7):
    
   
    lbp = feature.local_binary_pattern(image, 24, 8, method="uniform")
    
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
	# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
	# return the histogram of Local Binary Patterns
    return hist



def load_data(path_data, binn):

    """
    Load data from dataset in to ram
    """
    def load_file_from_folder(folder_name):

        X, y = [], []
        for folder in os.listdir(folder_name):

            folder_path = os.path.join(folder_name, folder)
            
            for filename in os.listdir(folder_path):

                file_path = os.path.join(folder_path, filename)
                try:
                    image = cv2.imread(file_path, 0)
                    # print(image.shape)
                    # image = cv2.cvtColor(image, cv2.COLOR_BRG2GRAY)
                 
                    if image is not None:
                        X.append(lbph(image, binn))
                        y.append(folder)
                except Exception as e:
                    print(e)
        return X, y

    X_train, y_train =  load_file_from_folder(path_data + '/' + 'train')
    X_valid, y_valid = load_file_from_folder(path_data + '/' + 'valid')

    return np.array(X_train), y_train, np.array(X_valid), y_valid