# 1. Importing necessary libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from autocorrect import spell
import matplotlib.pyplot as plt
from math import log, sqrt
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import numpy as np
import pandas as pd
import re

# 2. Loading the Dataset
messages = pd.read_csv('spam.csv', encoding = 'latin-1')
messages.head()

# 3.a. Dataset Pre-Processing
# removing unwanted columns
messages = messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
messages.head()

# renaming columns and converting to numerical form
messages = messages.rename(columns={'v1':'label', 'v2':'message'})
print(messages.label.value_counts())
messages['label'] = messages.label.map({'ham':0, 'spam':1})
messages.head()

# 4. Visualizing the Dataset
# basic description of the dataset
print('Dataset Description : ', messages.groupby('label').describe(),'\n')
# length of the messages 
messages['start_length'] = messages['message'].map(lambda msg: len(msg))
print(messages.head())

# spam messages word cloud
spamWords = ''.join(list(messages[messages['label']==1]['message']))
spamWordCloud = WordCloud(width=512, height=512).generate(spamWords)
plt.figure(figsize=(10,10), facecolor='k')
plt.imshow(spamWordCloud)
plt.show()

# ham messages word cloud
hamWords = ''.join(list(messages[messages['label']==0]['message']))
hamWordCloud = WordCloud(width=512, height=512).generate(hamWords)
plt.figure(figsize=(10,10), facecolor='k')
plt.imshow(hamWordCloud)
plt.show()

# 3.b. Dataset Preprocessing prep for training
def preprocessing(dataset):
    lbls = dataset['label']
    msgs = dataset['message']

    processedMessages = msgs.str.replace(r'[^A-Za-z\s\w\d\s+]', '')

    processedMessages = processedMessages.str.lower()

    stop_words = stopwords.words('english')
    processedMessages = processedMessages.apply(lambda words: ' '.join(word for word in words.split() if word not in set(stop_words)))

    stemmer = PorterStemmer()
    processedMessages = processedMessages.apply(lambda words: ' '.join(stemmer.stem(word) for word in words.split()))
    
    return pd.concat([lbls, processedMessages], axis=1)

processedMessages = preprocessing(messages)

processedMessages['after_length'] = processedMessages['message'].map(lambda msg: len(msg))
print(processedMessages.head())

#example = pd.DataFrame({'label':[0], 'message': ["""  ***** CONGRATlations **** You won 2 tIckETs to Hamilton in NYC http://www.hamiltonbroadway.com/J?NaIOl/event   wORtH over $500.00...CALL 555-477-8914 or send message to: hamilton@freetix.com to get ticket !! !  """]})
example = pd.DataFrame({'label':[1], 'message': ["Free entry in 2 a wkly comp to win FA Cup final"]})
processedExample = preprocessing(example)
print(processedExample)

# Message Lengths comparison before and after dataset prepeocessing
figure, ax = plt.subplots(1,2)
ax[0].hist(messages['start_length'], bins=50)
ax[0].set_title('Messages before Preprocessing')
ax[0].set_xlabel('message length')
ax[0].set_ylabel('frequency')

ax[1].hist(processedMessages['after_length'], bins=50)
ax[1].set_title('Messages after Preprocessing')
ax[1].set_xlabel('message length')
ax[1].set_ylabel('frequency')
plt.show()

# based on message length and label
messages['start_length'].hist(by = messages['label'])
processedMessages['after_length'].hist(by = processedMessages['label'])

# 5. Splitting dataset into Training and Testing 75:25 
print(processedMessages.head())
trainMessages, testMessages, trainLabels, testLabels = train_test_split(
    processedMessages["message"], processedMessages["label"], test_size = 0.25, random_state = 10
)
print(trainMessages.shape)
print(testMessages.shape)
print(trainLabels.shape)
print(testLabels.shape)

# 6. Classifier
vectorize = CountVectorizer(stop_words='english')
vectorize.fit(trainMessages)

trainMessagesDF = vectorize.transform(trainMessages)
testMessagesDF = vectorize.transform(testMessages)
exampleMessageDF = vectorize.transform(processedExample['message'])

# multinomial naive bayes classifier
mnbModel = MultinomialNB()
%time mnbModel.fit(trainMessagesDF,trainLabels)

mnbPrediction = mnbModel.predict(testMessagesDF)

print('Multinomial Naive Bayes F1 Score :', metrics.f1_score(testLabels, mnbPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, mnbPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('Multinomial Naive Bayes - Classification Report \n', classification_report(testLabels, mnbPrediction))

# bernoulli naive bayes classifier
bnbModel = BernoulliNB()
bnbModel.fit(trainMessagesDF,trainLabels)

bnbPrediction = bnbModel.predict(testMessagesDF)

print('Bernoulli Naive Bayes F1 Score :', metrics.f1_score(testLabels, bnbPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, bnbPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('Bernoulli Naive Bayes - Classification Report \n', classification_report(testLabels, bnbPrediction))

# linear svc classifier
linsvcModel = LinearSVC()
linsvcModel.fit(trainMessagesDF,trainLabels)

linsvcPrediction = linsvcModel.predict(testMessagesDF)

print('Linear SVC F1 Score :', metrics.f1_score(testLabels, linsvcPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, linsvcPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('Linear SVC - Classification Report \n', classification_report(testLabels, linsvcPrediction))

# svc classifier
svcModel = SVC(kernel='rbf')
svcModel.fit(trainMessagesDF,trainLabels)

svcPrediction = svcModel.predict(testMessagesDF)

print('SVC (rbf Kernel) F1 Score :', metrics.f1_score(testLabels, svcPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, svcPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('SVC (rbf Kernel) - Classification Report \n', classification_report(testLabels, svcPrediction))

# logistic regression classifier
logRegModel = LogisticRegression()
logRegModel.fit(trainMessagesDF, trainLabels)

logRegPrediction = logRegModel.predict(testMessagesDF)

print('Logistic Regression F1 Score :', metrics.f1_score(testLabels, logRegPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, logRegPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('Logistic Regression - Classification Report \n', classification_report(testLabels, logRegPrediction))

# decision tree classifier
dtcModel = DecisionTreeClassifier()
dtcModel.fit(trainMessagesDF, trainLabels)

dtcPrediction = dtcModel.predict(testMessagesDF)

print('Decision Tree F1 Score :', metrics.f1_score(testLabels, dtcPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, dtcPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('Decision Tree - Classification Report \n', classification_report(testLabels, dtcPrediction))

# K Nearest Neighbors classifier
kncModel = KNeighborsClassifier(1)
kncModel.fit(trainMessagesDF, trainLabels)

kncPrediction = kncModel.predict(testMessagesDF)

print('K Nearest Neighbors F1 Score :', metrics.f1_score(testLabels, kncPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, kncPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('K Nearest Neighbors - Classification Report \n', classification_report(testLabels, kncPrediction))

# random forest classifier
rfModel = RandomForestClassifier(n_estimators=1, min_samples_split=30)
rfModel.fit(trainMessagesDF,trainLabels)

rfPrediction = rfModel.predict(testMessagesDF)

print('Random Forest Classifier F1 Score :', metrics.f1_score(testLabels, rfPrediction))
# cross-validation using confusion matrix
pd.DataFrame(metrics.confusion_matrix(testLabels, rfPrediction),index=[['actual', 'actual'], ['spam', 'ham']], columns=[['predicted', 'predicted'], ['spam', 'ham']])

print('Random Forests - Classification Report \n', classification_report(testLabels, rfPrediction))

# 7. Model Performance Evaluation
scoring = {'accuracy':metrics.make_scorer(metrics.accuracy_score), 
           'precision':metrics.make_scorer(metrics.precision_score),
           'recall':metrics.make_scorer(metrics.recall_score), 
           'f1_score':metrics.make_scorer(metrics.f1_score)}

def modelPerformanceEvaluation(X, Y, cvFolds):
    MNB = cross_validate(mnbModel, X, Y, cv=cvFolds, scoring=scoring)
    BNB = cross_validate(bnbModel, X, Y, cv=cvFolds, scoring=scoring)
    linearSVC = cross_validate(linsvcModel, X, Y, cv=cvFolds, scoring=scoring)
    SVC = cross_validate(svcModel, X, Y, cv=cvFolds, scoring=scoring)
    logReg = cross_validate(logRegModel, X, Y, cv=cvFolds, scoring=scoring)
    DTC = cross_validate(dtcModel, X, Y, cv=cvFolds, scoring=scoring)
    KNC = cross_validate(kncModel, X, Y, cv=cvFolds, scoring=scoring)
    RF = cross_validate(rfModel, X, Y, cv=cvFolds, scoring=scoring)

    modelsScoreTable = pd.DataFrame(
        {'Multinomial Naive Bayes': [
            MNB['test_accuracy'].mean(),
            MNB['test_precision'].mean(),
            MNB['test_recall'].mean(),
            MNB['test_f1_score'].mean()
        ],
        'Bernoulli Naive Bayes': [
            BNB['test_accuracy'].mean(),
            BNB['test_precision'].mean(),
            BNB['test_recall'].mean(),
            BNB['test_f1_score'].mean()
        ],
        'Linear Support Vector Classifier': [
            linearSVC['test_accuracy'].mean(),
            linearSVC['test_precision'].mean(),
            linearSVC['test_recall'].mean(),
            linearSVC['test_f1_score'].mean()
        ],
        'Support Vector Classifier': [
            SVC['test_accuracy'].mean(),
            SVC['test_precision'].mean(),
            SVC['test_recall'].mean(),
            SVC['test_f1_score'].mean()
        ],
        'Logistic Regression': [
            logReg['test_accuracy'].mean(),
            logReg['test_precision'].mean(),
            logReg['test_recall'].mean(),
            logReg['test_f1_score'].mean()
        ],
        'Decision Tree Classifier': [
            DTC['test_accuracy'].mean(),
            DTC['test_precision'].mean(),
            DTC['test_recall'].mean(),
            DTC['test_f1_score'].mean()
        ],
        'K Nearest Neighbor Classifier': [
            KNC['test_accuracy'].mean(),
            KNC['test_precision'].mean(),
            KNC['test_recall'].mean(),
            KNC['test_f1_score'].mean()
        ],
        'Random Forest Classifier': [
            RF['test_accuracy'].mean(),
            RF['test_precision'].mean(),
            RF['test_recall'].mean(),
            RF['test_f1_score'].mean()
        ]},
        index = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    )

    modelsScoreTable['Best Score'] = modelsScoreTable.idxmax(axis=1)

    return modelsScoreTable

modelPerformanceEvaluation(trainMessagesDF, trainLabels, 8)

def roc_aucCurve(X, Y, testPrediction, classifier):
    spam_auc = roc_auc_score(testLabels, testPrediction)

    print(classifier, ' - Spam Messages: ROC AUC = %.3f' % (spam_auc))

    spamFalsePos, spamTruePos, _ = roc_curve(testLabels, testPrediction)
    return [spamFalsePos, spamTruePos]

res = roc_aucCurve(trainMessagesDF, trainLabels, mnbPrediction, 'Multinomial NB')
plt.plot(res[0], res[1], linestyle='--', color='orange', label='Multinomial NB')

res = roc_aucCurve(trainMessagesDF, trainLabels, bnbPrediction, 'Bernoulli NB')
plt.plot(res[0], res[1], linestyle='-', color='pink', label='Bernoulli NB')

res = roc_aucCurve(trainMessagesDF, trainLabels, linsvcPrediction, 'Linear SVC')
plt.plot(res[0], res[1], linestyle=':', color='green', label='Linear SVC')

res = roc_aucCurve(trainMessagesDF, trainLabels, svcPrediction, 'SVC (rbf Kernel)')
plt.plot(res[0], res[1], linestyle='-', color='purple', label='SVC')

res = roc_aucCurve(trainMessagesDF, trainLabels, logRegPrediction, 'Logistic Regression')
plt.plot(res[0], res[1], linestyle='dashdot', color='blue', label='Logistic Regression')

res = roc_aucCurve(trainMessagesDF, trainLabels, dtcPrediction, 'Decision Tree')
plt.plot(res[0], res[1], linestyle='--', color='black', label='Decision Tree')

def roc_aucCurve(X, Y, testPrediction, classifier):
    spam_auc = roc_auc_score(testLabels, testPrediction)

    print(classifier, ' - Spam Messages: ROC AUC = %.3f' % (spam_auc))

    spamFalsePos, spamTruePos, _ = roc_curve(testLabels, testPrediction)
    return [spamFalsePos, spamTruePos]

res = roc_aucCurve(trainMessagesDF, trainLabels, mnbPrediction, 'Multinomial NB')
plt.plot(res[0], res[1], linestyle='--', color='orange', label='Multinomial NB')

res = roc_aucCurve(trainMessagesDF, trainLabels, bnbPrediction, 'Bernoulli NB')
plt.plot(res[0], res[1], linestyle='-', color='pink', label='Bernoulli NB')

res = roc_aucCurve(trainMessagesDF, trainLabels, linsvcPrediction, 'Linear SVC')
plt.plot(res[0], res[1], linestyle=':', color='green', label='Linear SVC')

res = roc_aucCurve(trainMessagesDF, trainLabels, svcPrediction, 'SVC (rbf Kernel)')
plt.plot(res[0], res[1], linestyle='-', color='purple', label='SVC')

res = roc_aucCurve(trainMessagesDF, trainLabels, logRegPrediction, 'Logistic Regression')
plt.plot(res[0], res[1], linestyle='dashdot', color='blue', label='Logistic Regression')

res = roc_aucCurve(trainMessagesDF, trainLabels, dtcPrediction, 'Decision Tree')
plt.plot(res[0], res[1], linestyle='--', color='black', label='Decision Tree')

res = roc_aucCurve(trainMessagesDF, trainLabels, kncPrediction, 'K Nearest Neighbor')
plt.plot(res[0], res[1], linestyle=':', color='red', label='K Nearest Neighbor')

res = roc_aucCurve(trainMessagesDF, trainLabels, rfPrediction, 'Random Forest')
plt.plot(res[0], res[1], linestyle='dashdot', color='yellow', label='Random Forest')

plt.title('ROC Curve')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='best')

plt.show()

# Ensemble Classifier (Voting Classifier)
votingCLF = VotingClassifier(estimators=[
    ('MNB', mnbModel), ('BNB', bnbModel), ('linearSVC', linsvcModel), ('SVC', svcModel), ('logReg', logRegModel), ('DTC', dtcModel), ('KNC', kncModel), ('RF', rfModel)
], voting='hard')

for clf, label in zip([mnbModel, bnbModel, linsvcModel, svcModel, logRegModel, dtcModel, kncModel, rfModel, votingCLF], ['Multinomial NB', 'Bernoulli NB', 'Linear SVC', 'SVC (rfb Kernel', 'Logistic Regression', 'Decision Tree', 'K Nearest Neighbor', 'Random Forest', 'Ensemble']):
    scores = cross_val_score(clf, trainMessagesDF, trainLabels, scoring='accuracy', cv=9)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))