import pickle, time, cv2
from utils import load_data
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
from utils  import lbph

X_train, y_train, X_test, y_test = load_data('dataset', 14)

filename = 'model_saved/best_model_LBP_bin-14_acc-0.9259.pickle'
clf = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)

def predict(image_path):
    image_path = 'dataset/valid/0/1619136374.783259_Unknow.jpg'

    img = cv2.imread(image_path, 0)

    hist = lbph(img, 14)

    t1 = time.time()

    predict = clf.predict([hist])

    print(time.time() - t1)

    print(predict)


plot_confusion_matrix(clf, X_test, y_test) 

plt.savefig('confusion-matrix-lbp-14-points.png')
plt.show()

