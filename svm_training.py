from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from utils import load_data
import pickle

best = 0; binn = 0
for i in range(24, 32):
    X_train, y_train, X_test, y_test = load_data('dataset', i)

    clf = LinearSVC(C = 250).fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds.tolist())
    if acc > 0.92:
        best = acc 
        binn = i
        filename = 'model_saved/best_model_LBP_bin-{}_acc-{}.pickle'.format(i, best)
        pickle.dump(clf, open(filename, 'wb'))
        
    # print ("Accuracy for validation : %.2f %%" %(100*acc) ) 

# 0,82 with 10
print(best, binn)