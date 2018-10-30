import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")


vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,3),dtype=np.float32)


tr_vect = vect_word.fit_transform(train['comment_text'])
ts_vect = vect_word.transform(test['comment_text'])

X = tr_vect
x_test = ts_vect

target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]

prd = np.zeros((x_test.shape[0],y.shape[1]))
cv_score =[]

for i,col in enumerate(target_col):
    lr = LogisticRegression(C=2,random_state = i,class_weight = 'balanced')
    print('Building {} model for column:{''}'.format(i,col))
    lr.fit(X,y[col])
    prd[:,i] = lr.predict_proba(x_test)[:,1]

#for col in target_col:
#	print("Column:",col)
#	pred =  lr.predict(X)
#	print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
#	print(classification_report(y[col],pred))

#for col in target_col:
#	print("Column:",col)
#	pred_pro = lr.predict_proba(X)[:,1]
#	frp,trp,thres = roc_curve(y[col],pred_pro)
#	print(auc(frp,trp))


#prd_1 = pd.DataFrame(prd,columns=y.columns)
#submit = pd.concat([test['id'],prd_1],axis=1)
#submit.to_csv('input/submission.csv',index=False)
#submit.head()
