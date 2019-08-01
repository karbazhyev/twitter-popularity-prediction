"""Problem 6"""
import numpy as np
import json
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

#%%=============================================================================
#Data Preprocessing
#===============================================================================
hashtag = '#superbowl'
subStrWA = ['WA', 'Washington']
subStrMA = ['MA', 'Massachusetts']
data = []
label = []
with open('tweet_data/tweets_%s.txt' % hashtag, 'r') as tweetFile:
    print('Hashtag: %s File Loaded' % hashtag)
    for tweetLine in tweetFile:
        tweet = json.loads(tweetLine)
        location = unicodedata.normalize('NFKD', tweet['tweet']['user']['location']).encode('ascii', 'ignore')
        for subStr in subStrWA:
            if subStr in location:
                label.append(-1)
                txt = unicodedata.normalize('NFKD', tweet['tweet']['text']).encode('ascii', 'ignore')
                data.append(txt)
                break
        for subStr in subStrMA:
            if subStr in location:
                label.append(1)
                txt = unicodedata.normalize('NFKD', tweet['tweet']['text']).encode('ascii', 'ignore')
                data.append(txt)
                break

numOfData = len(data)
indices = np.arange(numOfData)
np.random.shuffle(indices)

indicesTest = indices[range(numOfData/10)]
dataTest = [data[i] for i in indicesTest]
labelTest = [label[i] for i in indicesTest]

indicesTrain = np.delete(indices, range(numOfData/10))
dataTrain = [data[i] for i in indicesTrain]
labelTrain = [label[i] for i in indicesTrain]

#%%=============================================================================

TFxIDF_vect = TfidfVectorizer(stop_words='english',  min_df=0.0001)
X_train_TFxIDF = TFxIDF_vect.fit_transform(dataTrain)
vocab = TFxIDF_vect.vocabulary_
X_train_TFxIDF = X_train_TFxIDF.transpose()
print(X_train_TFxIDF.shape)

# LSI
u, s, vt = svds(X_train_TFxIDF, k=50)
ut = u.T
idx = np.floor(np.linspace(0, X_train_TFxIDF.shape[1], 50))
for ii in range(idx.size-1):
    temp = (X_train_TFxIDF[:,idx[ii]:idx[ii+1]].toarray())
    if ii == 0:
        Dk_train = ut.dot(temp)
    else:
        Dk_train = np.c_[Dk_train, ut.dot(temp)]

TFxIDF_vect = TfidfVectorizer(vocabulary=vocab)
X_test_TFxIDF = TFxIDF_vect.fit_transform(dataTest)
X_test_TFxIDF = X_test_TFxIDF.transpose()
idx = np.floor(np.linspace(0, X_test_TFxIDF.shape[1], 50))
for ii in range(idx.size-1):
    temp = (X_test_TFxIDF[:,idx[ii]:idx[ii+1]].toarray())
    if ii == 0:
        Dk_test = ut.dot(temp)
    else:
        Dk_test = np.c_[Dk_test, ut.dot(temp)]

# %%=============================================================================
# SVM-Hard Margin
# ===============================================================================
from sklearn import svm
from sklearn.metrics import confusion_matrix

ROC_points = 200

class_train = np.copy(labelTrain)
class_test = np.copy(labelTest)

clf_HM = svm.SVC(C=1000, kernel='linear', shrinking=False)

clf_HM.fit(Dk_train.T, class_train)
print(clf_HM.score(Dk_test.T, class_test))

w_HM = clf_HM.coef_
b_HM = clf_HM.intercept_
my_class_train = np.sign(w_HM.dot(Dk_train) + b_HM)

predict_test = clf_HM.predict(Dk_test.T)
confusion_matrix_HM = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Hard Margin: ', confusion_matrix_HM)
print('Precision: ', float(confusion_matrix_HM[0, 0]) / sum(confusion_matrix_HM[:, 0]), 'Accuracy: ',
      (float(confusion_matrix_HM[0, 0] + confusion_matrix_HM[1, 1])) / np.sum(confusion_matrix_HM.flatten()),
      'Recall: ', float(confusion_matrix_HM[0, 0]) / np.sum(confusion_matrix_HM[0, :]))

b1 = np.linspace(-10, 10, ROC_points)
b1 = b1 + b_HM
test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
test_negatives = (class_test < 0)
test_negatives_num = sum(test_negatives)
false_positive_rate_HM = np.zeros(ROC_points)
true_positive_rate_HM = np.zeros(ROC_points)
for ii in range(len(b1)):
    classes = np.sign(w_HM.dot(Dk_test) + b1[ii])[0]
    positives = (classes > 0)
    true_positives_HM = np.logical_and(positives, test_positives)
    false_positives_HM = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_HM[ii] = np.sum(true_positives_HM) / float(test_positives_num)
    false_positive_rate_HM[ii] = np.sum(false_positives_HM) / float(test_negatives_num)

plt.figure()
plt.plot(false_positive_rate_HM, true_positive_rate_HM, linewidth=2)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Hard Margin SVM', fontsize=20)

# %%============================================================================
# Naive Bayes
# ==============================================================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

ROC_points = 200

class_train = np.copy(labelTrain)
class_test = np.copy(labelTest)

clf_NB = GaussianNB()
clf_NB.fit(Dk_train.T, class_train)
# clf_NB.fit(X_train_TFxIDF.transpose(), class_train)

# predict_test = clf_NB.predict(X_test_TFxIDF.transpose())
#predict_test = clf_NB.predict(Dk_test.T, class_test)
predict_test = clf_NB.predict(Dk_test.T)
confusion_matrix_NB = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Naive Bayes: ', confusion_matrix_NB)
print('Precision: ', float(confusion_matrix_NB[0, 0]) / sum(confusion_matrix_NB[:, 0]), 'Accuracy: ',
      float(confusion_matrix_NB[0, 0] + confusion_matrix_NB[1, 1]) / np.sum(confusion_matrix_NB.flatten()),
      'Recall: ', float(confusion_matrix_NB[0, 0]) / np.sum(confusion_matrix_NB[0, :]))

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
test_negatives = (class_test < 0)
test_negatives_num = sum(test_negatives)
false_positive_rate_NB = np.zeros(ROC_points)
true_positive_rate_NB = np.zeros(ROC_points)
# class_probs = clf_NB.predict_proba(X_test_TFxIDF.transpose())
class_probs = clf_NB.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    positives = (class_probs[:, 0] < p1[ii])
    true_positives_NB = np.logical_and(positives, test_positives)
    false_positives_NB = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_NB[ii] = np.sum(true_positives_NB) / float(test_positives_num)
    false_positive_rate_NB[ii] = np.sum(false_positives_NB) / float(test_negatives_num)

plt.figure()
plt.plot(false_positive_rate_NB, true_positive_rate_NB, linewidth=2)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Naive Bayes', fontsize=20)

# %%============================================================================
# Logistic Regression Classifier
# ==============================================================================
from sklearn.linear_model import LogisticRegression

ROC_points = 200

class_train = np.copy(labelTrain)
class_test = np.copy(labelTest)

clf_LG = LogisticRegression(C=1000)

clf_LG.fit(Dk_train.T, class_train)

predict_test = clf_LG.predict(Dk_test.T)
confusion_matrix_LG = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Logistic Regression: ', confusion_matrix_LG)
print('Precision: ', float(confusion_matrix_LG[0, 0]) / sum(confusion_matrix_LG[:, 0]), 'Accuracy: ',
      float(confusion_matrix_LG[0, 0] + confusion_matrix_LG[1, 1]) / np.sum(confusion_matrix_LG.flatten()),
      'Recall: ', float(confusion_matrix_LG[0, 0]) / np.sum(confusion_matrix_LG[0, :]))

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
test_negatives = (class_test < 0)
test_negatives_num = sum(test_negatives)
false_positive_rate_LG = np.zeros(ROC_points)
true_positive_rate_LG = np.zeros(ROC_points)
class_probs = clf_LG.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    positives = (class_probs[:, 0] < p1[ii])
    true_positives_LG = np.logical_and(positives, test_positives)
    false_positives_LG = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_LG[ii] = np.sum(true_positives_LG) / float(test_positives_num)
    false_positive_rate_LG[ii] = np.sum(false_positives_LG) / float(test_negatives_num)

plt.figure()
plt.plot(false_positive_rate_LG, true_positive_rate_LG, linewidth=2)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positve Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Logistic Regression', fontsize=20)

# %%============================================================================
# Logistic Regression with L1 regularizer
# ==============================================================================
from sklearn.linear_model import LogisticRegression

ROC_points = 200

class_train = np.copy(labelTrain)
class_test = np.copy(labelTest)

clf_LG_L1 = LogisticRegression(penalty='l1', C=1)

clf_LG_L1.fit(Dk_train.T, class_train)
predict_test = clf_LG_L1.predict(Dk_test.T)
confusion_matrix_LG_L1 = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Soft Margin Logistic Regression with L1 Penalty: ', confusion_matrix_LG_L1)
print('Precision: ', float(confusion_matrix_LG_L1[0, 0]) / sum(confusion_matrix_LG_L1[:, 0]), 'Accuracy: ',
      float(confusion_matrix_LG_L1[0, 0] + confusion_matrix_LG_L1[1, 1]) / np.sum(confusion_matrix_LG_L1.flatten()),
      'Recall: ', float(confusion_matrix_LG_L1[0, 0]) / np.sum(confusion_matrix_LG_L1[0, :]))

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
test_negatives = (class_test < 0)
test_negatives_num = sum(test_negatives)
false_positive_rate_LG_L1 = np.zeros(ROC_points)
true_positive_rate_LG_L1 = np.zeros(ROC_points)
class_probs = clf_LG_L1.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    positives = (class_probs[:, 0] < p1[ii])
    true_positives_LG_L1 = np.logical_and(positives, test_positives)
    false_positives_LG_L1 = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_LG_L1[ii] = np.sum(true_positives_LG_L1) / float(test_positives_num)
    false_positive_rate_LG_L1[ii] = np.sum(false_positives_LG_L1) / float(test_negatives_num)

plt.figure()
plt.plot(false_positive_rate_LG_L1, true_positive_rate_LG_L1, linewidth=2)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Logistic Regression with L1 regularization', fontsize=20)

# %%============================================================================
# Logistic Regression with L2 regularizer
# ==============================================================================
from sklearn.linear_model import LogisticRegression

ROC_points = 200

class_train = np.copy(labelTrain)
class_test = np.copy(labelTest)

clf_LG_L2 = LogisticRegression(penalty='l2', C=1)

clf_LG_L2.fit(Dk_train.T, class_train)
predict_test = clf_LG_L2.predict(Dk_test.T)
confusion_matrix_LG_L2 = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Logistic Regression with L2 Penalty: ', confusion_matrix_LG_L2)
print('Precision: ', float(confusion_matrix_LG_L2[0, 0]) / sum(confusion_matrix_LG_L2[:, 0]), 'Accuracy: ',
      float(confusion_matrix_LG_L2[0, 0] + confusion_matrix_LG_L2[1, 1]) / np.sum(confusion_matrix_LG_L2.flatten()),
      'Recall: ', float(confusion_matrix_LG_L2[0, 0]) / np.sum(confusion_matrix_LG_L2[0, :]))


p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
test_negatives = (class_test < 0)
test_negatives_num = sum(test_negatives)
false_positive_rate_LG_L2 = np.zeros(ROC_points)
true_positive_rate_LG_L2 = np.zeros(ROC_points)
class_probs = clf_LG_L2.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    positives = (class_probs[:, 0] < p1[ii])
    true_positives_LG_L2 = np.logical_and(positives, test_positives)
    false_positives_LG_L2 = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_LG_L2[ii] = np.sum(true_positives_LG_L2) / float(test_positives_num)
    false_positive_rate_LG_L2[ii] = np.sum(false_positives_LG_L2) / float(test_negatives_num)

plt.figure()
plt.plot(false_positive_rate_LG_L2, true_positive_rate_LG_L2, linewidth=2)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Logistic Regression with L2 regularization', fontsize=20)