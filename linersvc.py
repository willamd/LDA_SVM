import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation,NMF
from sklearn.svm import SVC,LinearSVC
from pandas import HDFStore, read_hdf
from sklearn.utils import Bunch
from sklearn.metrics import f1_score,accuracy_score,label_ranking_average_precision_score,mean_squared_error
from time import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


def type2idx(Data_c,Type_c):
    n_samples=len(Data_c)
    target = np.empty((n_samples,), dtype=np.int)
    for idx in range(n_samples):
        if Data_c[idx] in Type_c:
            target[idx]=Type_c.index(Data_c[idx])
        else:
            target[idx] = -1
    return target
# All_data=read_hdf('./data/RandomSplittedByCatagories.h5',key='AllData')
#
# data=list(All_data['Service Desciption'])
# target=list(All_data['Service Classification'])
#
# All_data=Bunch(data=data,target=target)
# X, Y = shuffle(All_data.data, All_data.target, random_state=1)
#
# print("Loading dataset...")
# # n_components = 10
# n_top_words = 20
# # data_samples = data_train.data[:n_samples]

#
# offset = int(len(X) * 0.5)
# X_train, Y_train = X[:offset], Y[:offset]
# X_test, Y_test = X[offset:], Y[offset:]
TrainServices = read_hdf('./data/RandomSplittedByCatagories.h5', key='Train')
TestServices = read_hdf('./data/RandomSplittedByCatagories.h5', key='Test')
# All_data=read_hdf('./data/RandomSplittedByCatagories.h5',key='AllData')

data_train=list(TrainServices['Service Desciption'])
target_train=list(TrainServices['Service Classification'])
data_test=list(TestServices['Service Desciption'])
target_test=list(TestServices['Service Classification'])

# Train_data=Bunch(data=data_train,target=target_train)
# Test_data=Bunch(data=data_test,target=target_test)
# X, Y = shuffle(All_data.data, All_data.target, random_state=13)

X_train=data_train
Y_train=target_train
X_test=data_test
Y_test=target_test
Type_c=(list(np.unique(Y_train)))
Y_train=type2idx(Y_train,Type_c)
Y_test=type2idx(Y_test,Type_c)

n_samples=len(X_train)
print("Service Description: \n" ,X_train[0])
print("Service Classification:",Y_train[0])
print(len(X_train))
print(len(X_test))

# Use tf-idf features for LDA.
print("Extracting tf-idf features for LDA...")
tfidf_vectorizer=TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=1500)

X_train = tfidf_vectorizer.fit_transform(X_train)

lda = LatentDirichletAllocation(n_topics=275,perp_tol=1e-4,mean_change_tol=1e-4,
                                        max_iter=10,
                                        learning_method='batch', random_state=0, n_jobs=1, evaluate_every=1,
                                        verbose=0)

X_train=lda.fit_transform(X_train,Y_train)

print(X_train[0])
# 打印最佳模型
# joblib.dump(lda, 'lda.model')
# print("\nTopics in LDA model:")

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print_top_words(lda, tfidf_feature_names, n_top_words)

# tf_train = tfidf_vectorizer.fit_transform(data_train.data)


# lda = joblib.load('lda.model')

X_test = tfidf_vectorizer.transform(X_test)
X_test=lda.transform(X_test)
print(X_test[0])
# lda_Test = lda.transform(tf_test)

# train_target=y_train
# test_target=y_test
max_iter=range(1,100,1)
plot_data=[]
train_errors1 = list()
test_errors1 = list()
train_errorstop5 = list()
test_errorstop5 = list()

train_acctop1=list()
test_acctop1=list()
train_acctop5=list()
test_acctop5=list()
# model=svm_cross_validation(ldax,train_target)
# print("Model Done")
# joblib.dump(model, 'svm.md')
# model = joblib.load('svm.md')

for idx, iter in enumerate(max_iter):
    # clf=SVC(gamma=1,probability=True,decision_function_shape='ovo',kernel='rbf',max_iter=iter,tol=1e-5)
    clf = LinearSVC(penalty='l2', dual=False,tol=1e-3,max_iter=iter)
    t0 = time()
    clf.fit(X_train,Y_train)
    t1 = time()
    print("Time:",t1-t0)
    # train_top5=clf.predict_proba(X_train)
    train_top1=clf.predict(X_train)

    train_acctop1.append((iter,accuracy_score(Y_train, train_top1)))

    train_errors1.append((iter, mean_squared_error(Y_train, train_top1)))

    # print ('acc：%.2f%' %  100*acc)
    # tfidf_vectorizer = TfidfVectorizer(#max_df=0.75, min_df=5,
    #                                    max_features=110,
    #                                    stop_words='english')
    # X_test = tfidf_vectorizer.fit_transform(X_test)
    # tf_test = tfidf_vectorizer.fit_transform(data_test.data)
    # ldax_test=text2features(data_test,n_top_words)
    # lda = joblib.load('lda.model')
    # ldax_test = lda.transform(tf_test)
    # test_target=type2idx(Test_Y,Type_c)
    test_pre_top1 = clf.predict(X_test)
    # test_pre_top5 = clf.predict_proba(X_test)
    test_acctop1.append((iter,accuracy_score(Y_test, test_pre_top1)))
    test_errors1.append((iter, mean_squared_error(Y_test, test_pre_top1)))
    # ret=np.empty((len(Test_Y),), dtype=np.int)
    # train_ret=np.empty((len(Train_Y),), dtype=np.int)

    f1_s=f1_score(Y_test, test_pre_top1, average='micro')
    train_acctop5.append((iter, accuracy_score(Y_train, train_top1)))
    test_acctop5.append((iter, accuracy_score(Y_test, test_pre_top1)))
    print("Test acc:%f,train acc:%f"%(accuracy_score(Y_test, test_pre_top1),accuracy_score(Y_train, train_top1)))
    print("Epoch:%d,F1_score:%f"%(iter,float(f1_s)))
    plot_data.append((iter,f1_s))

# print(f1_score(train_target, train_ret, average='micro'))

joblib.dump(plot_data, 'result_Lsvc.dat')
joblib.dump(train_errors1, 'train_errors1_Lsv.dat')
joblib.dump(test_errors1, 'test_errors1_Lsv.dat')
joblib.dump(train_acctop1, 'train_acctop1_Lsv.dat')
joblib.dump(train_acctop1, 'test_acctop1_Lsv.dat')
# joblib.dump(train_acctop5, 'train_acctop5_Lsv.dat')
# joblib.dump(test_acctop5, 'test_acctop5_Lsv.dat')
# joblib.dump(test_errorstop5, 'test_errorstop5_Lsv.dat')
# joblib.dump(train_errorstop5, 'train_errorstop5_Lsv.dat')
plt.plot(*zip(*plot_data))