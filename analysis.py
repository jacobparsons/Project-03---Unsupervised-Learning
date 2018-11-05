import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition.pca import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm
 
digits = load_digits()
cancer = load_breast_cancer()

#Digits Dimensionality Reduction
compressor = PCA()
compressor.fit(digits.data)
pca_digits_data = compressor.transform(digits.data)

compressor = FastICA()
compressor.fit(digits.data)
ica_digits_data = compressor.transform(digits.data)

compressor = RandomProjection(64)
compressor.fit(digits.data)
randproj_digits_data = compressor.transform(digits.data)

compressor = best(chi2)
compressor.fit(digits.data,digits.target)
kbest_digits_data = compressor.transform(digits.data)

#Cancer Dimensionality Reduction
compressor = PCA()
compressor.fit(cancer.data)
pca_cancer_data = compressor.transform(cancer.data)

compressor = FastICA()
compressor.fit(cancer.data)
ica_cancer_data = compressor.transform(cancer.data)

compressor = RandomProjection(64)
compressor.fit(cancer.data)
randproj_cancer_data = compressor.transform(cancer.data)

compressor = best(chi2)
compressor.fit(cancer.data,cancer.target)
kbest_cancer_data = compressor.transform(cancer.data)

#Digits KMeans
digits_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(digits.data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(digits.data)):
        datapoint = digits.data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    digits_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
digits_scores_df = pd.DataFrame(digits_scores)
plt.plot(digits_scores_df[0],digits_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Digits: KMeans')

plt.savefig('kmeans_digits.png')
plt.clf()

#





#
#X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.33,random_state=50)
#

#Digits PCA KMeans
digits_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(pca_digits_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(pca_digits_data)):
        datapoint = pca_digits_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    digits_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
digits_scores_df = pd.DataFrame(digits_scores)
plt.plot(digits_scores_df[0],digits_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Digits: KMeans and PCA')

plt.savefig('kmeans_pca_digits.png')
plt.clf()

#Digits ICA KMeans
digits_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(ica_digits_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(ica_digits_data)):
        datapoint = ica_digits_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    digits_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
digits_scores_df = pd.DataFrame(digits_scores)
plt.plot(digits_scores_df[0],digits_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Digits: KMeans and ICA')

plt.savefig('kmeans_ica_digits.png')
plt.clf()


#Digits Random Projection KMeans
digits_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(randproj_digits_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(randproj_digits_data)):
        datapoint = randproj_digits_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    digits_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
digits_scores_df = pd.DataFrame(digits_scores)
plt.plot(digits_scores_df[0],digits_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Digits: KMeans and Random Projection')

plt.savefig('kmeans_randproj_digits.png')
plt.clf()



#Digits Random KBest KMeans
digits_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(kbest_digits_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(kbest_digits_data)):
        datapoint = kbest_digits_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    digits_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
digits_scores_df = pd.DataFrame(digits_scores)
plt.plot(digits_scores_df[0],digits_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Digits: KMeans and KBest')

plt.savefig('kmeans_kbest_digits.png')
plt.clf()


#Cancer KMeans
cancer_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(cancer.data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(cancer.data)):
        datapoint = cancer.data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    cancer_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
cancer_scores_df = pd.DataFrame(cancer_scores)
plt.plot(cancer_scores_df[0],cancer_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Cancer: KMeans')

plt.savefig('kmeans_cancer.png')
plt.clf()


X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,test_size=0.33,random_state=50)


#Cancer PCA KMeans
cancer_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(pca_cancer_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(pca_cancer_data)):
        datapoint = pca_cancer_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    cancer_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
cancer_scores_df = pd.DataFrame(cancer_scores)
plt.plot(cancer_scores_df[0],cancer_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Cancer: KMeans and PCA')

plt.savefig('kmeans_pca_cancer.png')
plt.clf()


#Cancer ICA KMeans
cancer_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(ica_cancer_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(ica_cancer_data)):
        datapoint = ica_cancer_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    cancer_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
cancer_scores_df = pd.DataFrame(cancer_scores)
plt.plot(cancer_scores_df[0],cancer_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Cancer: KMeans and ICA')

plt.savefig('kmeans_ica_cancer.png')
plt.clf()


#Cancer Random Projection KMeans
cancer_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(randproj_cancer_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(randproj_cancer_data)):
        datapoint = randproj_cancer_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    cancer_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
cancer_scores_df = pd.DataFrame(cancer_scores)
plt.plot(cancer_scores_df[0],cancer_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Cancer: KMeans and Random Projection')

plt.savefig('kmeans_randproj_cancer.png')
plt.clf()


#Cancer Random KBest KMeans
cancer_scores = []
for cluster_count in range(1,15):
    classifier = KMeans(n_clusters=cluster_count,random_state=0)
    classifier.fit(kbest_cancer_data)
   
    labels = classifier.labels_
    centers = classifier.cluster_centers_
   
    distances = []
    for i in range(0,len(kbest_cancer_data)):
        datapoint = kbest_cancer_data[i]
        center = centers[labels[i]]
        distance = norm(datapoint-center)
        distances.append(distance)
    avg_distance = sum(distances)/len(distances)
    cancer_scores.append([cluster_count,avg_distance])
    print cluster_count, avg_distance
   
cancer_scores_df = pd.DataFrame(cancer_scores)
plt.plot(cancer_scores_df[0],cancer_scores_df[1])
plt.xlabel('Cluster Count')
plt.ylabel('Avg Distance to Cluster Center')
plt.title('Cancer: KMeans and KBest')

plt.savefig('kmeans_kbest_cancer.png')
plt.clf()





#Digits EM
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=10,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
cluster_map = [test_df[test_df.label == i]['y_train'].value_counts().idxmax() for i in range(0,10)]
test_df['label_adj'] = test_df['label'].apply(lambda x: cluster_map[x])
test_df['Error'] = test_df['y_train'] != test_df['label_adj']
digits_error = float(sum(test_df['Error']))/float(len(test_df['Error']))

#Digits PCA EM
X_train, X_test, y_train, y_test = train_test_split(pca_digits_data,digits.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=10,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
cluster_map = [test_df[test_df.label == i]['y_train'].value_counts().idxmax() for i in range(0,10)]
test_df['label_adj'] = test_df['label'].apply(lambda x: cluster_map[x])
test_df['Error'] = test_df['y_train'] != test_df['label_adj']
pca_digits_error = float(sum(test_df['Error']))/float(len(test_df['Error']))

#Digits ICA EM
X_train, X_test, y_train, y_test = train_test_split(ica_digits_data,digits.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=10,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
cluster_map = [test_df[test_df.label == i]['y_train'].value_counts().idxmax() for i in range(0,10)]
test_df['label_adj'] = test_df['label'].apply(lambda x: cluster_map[x])
test_df['Error'] = test_df['y_train'] != test_df['label_adj']
ica_digits_error = float(sum(test_df['Error']))/float(len(test_df['Error']))

#Digits Random Projection EM
X_train, X_test, y_train, y_test = train_test_split(randproj_digits_data,digits.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=10,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
cluster_map = [test_df[test_df.label == i]['y_train'].value_counts().idxmax() for i in range(0,10)]
test_df['label_adj'] = test_df['label'].apply(lambda x: cluster_map[x])
test_df['Error'] = test_df['y_train'] != test_df['label_adj']
randproj_digits_error = float(sum(test_df['Error']))/float(len(test_df['Error']))


#Digits KBest EM
X_train, X_test, y_train, y_test = train_test_split(kbest_digits_data,digits.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=10,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
cluster_map = [test_df[test_df.label == i]['y_train'].value_counts().idxmax() for i in range(0,10)]
test_df['label_adj'] = test_df['label'].apply(lambda x: cluster_map[x])
test_df['Error'] = test_df['y_train'] != test_df['label_adj']
kbest_digits_error = float(sum(test_df['Error']))/float(len(test_df['Error']))


objects = ('No \nTransformation', 'PCA', 'ICA', 'Random \nProjection', 'KBest')
y_pos = np.arange(len(objects))
performance = [digits_error,pca_digits_error,ica_digits_error,randproj_digits_error,kbest_digits_error]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Error')
plt.title('Digits: EM Performance')
 
plt.savefig('em_error_digits.png')
plt.clf()






#Cancer EM
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=2,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
test_df['Error'] = test_df['y_train'] != test_df['label']
cancer_error = float(sum(test_df['Error']))/float(len(test_df['Error']))

#Cancer PCA EM
X_train, X_test, y_train, y_test = train_test_split(pca_cancer_data,cancer.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=2,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
test_df['Error'] = test_df['y_train'] != test_df['label']
pca_cancer_error = float(sum(test_df['Error']))/float(len(test_df['Error']))

#Cancer ICA EM
X_train, X_test, y_train, y_test = train_test_split(ica_cancer_data,cancer.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=2,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
test_df['Error'] = test_df['y_train'] != test_df['label']
ica_cancer_error = float(sum(test_df['Error']))/float(len(test_df['Error']))

#Cancer Random Projection EM
X_train, X_test, y_train, y_test = train_test_split(randproj_cancer_data,cancer.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=2,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
test_df['Error'] = test_df['y_train'] != test_df['label']
randproj_cancer_error = float(sum(test_df['Error']))/float(len(test_df['Error']))


#Cancer KBest EM
X_train, X_test, y_train, y_test = train_test_split(kbest_cancer_data,cancer.target,test_size=0.33,random_state=50)
classifier = GMM(n_components=2,random_state=0)
classifier.fit(X_train)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
test_df['label'] = classifier.predict(X_train)
test_df['y_train'] = y_train
test_df['Error'] = test_df['y_train'] != test_df['label']
kbest_cancer_error = float(sum(test_df['Error']))/float(len(test_df['Error']))


objects = ('No \nTransformation', 'PCA', 'ICA', 'Random \nProjection', 'KBest')
y_pos = np.arange(len(objects))
performance = [cancer_error,pca_cancer_error,ica_cancer_error,randproj_cancer_error,kbest_cancer_error]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Error')
plt.title('Cancer: EM Performance')
 
plt.savefig('em_error_cancer.png')
plt.clf()






#Digits Neural Network Unaltered Data
scores_nn_df = pd.DataFrame(columns=['neuron_count','train_score','test_score','crossval_score'])
for neuron_count in range(2,100,2):
    X_train_restricted, X_test_restricted, y_train_restricted, y_test_restricted = train_test_split(digits.data,digits.target,test_size=0.33,random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=neuron_count,activation='logistic')
    classifier.fit(X_train_restricted,y_train_restricted)
    train_score = classifier.score(X_train_restricted,y_train_restricted)
    test_score = classifier.score(X_test_restricted,y_test_restricted)
    try:
        cv_score = cross_val_score(classifier,X_train_restricted,y_train_restricted,cv=5).mean()
    except:
        cv_score = np.nan
    scores_nn_df.loc[neuron_count] = [neuron_count,train_score,test_score,cv_score]
   
    print "neuron_count: "+str(neuron_count)+" "+str(cv_score)

plt.plot(scores_nn_df['neuron_count'],scores_nn_df['train_score'],label='train')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['test_score'],label='test')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['crossval_score'],label='crossval')
plt.xlabel('Neuron Count')
plt.ylabel('Score')
plt.title('Neural Network')
plt.legend()

plt.savefig('digits_nn_learning_curve.png')
plt.clf()
 
 
#Digits Neural Network PCA Data
scores_nn_df = pd.DataFrame(columns=['neuron_count','train_score','test_score','crossval_score'])
for neuron_count in range(2,100,2):
    X_train_restricted, X_test_restricted, y_train_restricted, y_test_restricted = train_test_split(pca_digits_data,digits.target,test_size=0.33,random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=neuron_count,activation='logistic')
    classifier.fit(X_train_restricted,y_train_restricted)
    train_score = classifier.score(X_train_restricted,y_train_restricted)
    test_score = classifier.score(X_test_restricted,y_test_restricted)
    try:
        cv_score = cross_val_score(classifier,X_train_restricted,y_train_restricted,cv=5).mean()
    except:
        cv_score = np.nan
    scores_nn_df.loc[neuron_count] = [neuron_count,train_score,test_score,cv_score]
   
    print "neuron_count: "+str(neuron_count)+" "+str(cv_score)

plt.plot(scores_nn_df['neuron_count'],scores_nn_df['train_score'],label='train')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['test_score'],label='test')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['crossval_score'],label='crossval')
plt.xlabel('Neuron Count')
plt.ylabel('Score')
plt.title('Digits: Neural Network and PCA')
plt.legend()

plt.savefig('nn_pca_learning_curve_digits.png')
plt.clf()
 
 
 
#Digits Neural Network ICA Data
scores_nn_df = pd.DataFrame(columns=['neuron_count','train_score','test_score','crossval_score'])
for neuron_count in range(2,100,2):
    X_train_restricted, X_test_restricted, y_train_restricted, y_test_restricted = train_test_split(ica_digits_data,digits.target,test_size=0.33,random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=neuron_count,activation='logistic')
    classifier.fit(X_train_restricted,y_train_restricted)
    train_score = classifier.score(X_train_restricted,y_train_restricted)
    test_score = classifier.score(X_test_restricted,y_test_restricted)
    try:
        cv_score = cross_val_score(classifier,X_train_restricted,y_train_restricted,cv=5).mean()
    except:
        cv_score = np.nan
    scores_nn_df.loc[neuron_count] = [neuron_count,train_score,test_score,cv_score]
   
    print "neuron_count: "+str(neuron_count)+" "+str(cv_score)

plt.plot(scores_nn_df['neuron_count'],scores_nn_df['train_score'],label='train')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['test_score'],label='test')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['crossval_score'],label='crossval')
plt.xlabel('Neuron Count')
plt.ylabel('Score')
plt.title('Digits: Neural Network and ICA')
plt.legend()

plt.savefig('nn_ica_learning_curve_digits.png')
plt.clf()
 

#Digits Neural Network Random Projection Data
scores_nn_df = pd.DataFrame(columns=['neuron_count','train_score','test_score','crossval_score'])
for neuron_count in range(2,100,2):
    X_train_restricted, X_test_restricted, y_train_restricted, y_test_restricted = train_test_split(randproj_digits_data,digits.target,test_size=0.33,random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=neuron_count,activation='logistic')
    classifier.fit(X_train_restricted,y_train_restricted)
    train_score = classifier.score(X_train_restricted,y_train_restricted)
    test_score = classifier.score(X_test_restricted,y_test_restricted)
    try:
        cv_score = cross_val_score(classifier,X_train_restricted,y_train_restricted,cv=5).mean()
    except:
        cv_score = np.nan
    scores_nn_df.loc[neuron_count] = [neuron_count,train_score,test_score,cv_score]
   
    print "neuron_count: "+str(neuron_count)+" "+str(cv_score)

plt.plot(scores_nn_df['neuron_count'],scores_nn_df['train_score'],label='train')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['test_score'],label='test')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['crossval_score'],label='crossval')
plt.xlabel('Neuron Count')
plt.ylabel('Score')
plt.title('Digits: Neural Network and Random Projection')
plt.legend()

plt.savefig('nn_randproj_learning_curve_digits.png')
plt.clf()


#Digits Neural Network KBest
scores_nn_df = pd.DataFrame(columns=['neuron_count','train_score','test_score','crossval_score'])
for neuron_count in range(2,100,2):
    X_train_restricted, X_test_restricted, y_train_restricted, y_test_restricted = train_test_split(kbest_digits_data,digits.target,test_size=0.33,random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=neuron_count,activation='logistic')
    classifier.fit(X_train_restricted,y_train_restricted)
    train_score = classifier.score(X_train_restricted,y_train_restricted)
    test_score = classifier.score(X_test_restricted,y_test_restricted)
    try:
        cv_score = cross_val_score(classifier,X_train_restricted,y_train_restricted,cv=5).mean()
    except:
        cv_score = np.nan
    scores_nn_df.loc[neuron_count] = [neuron_count,train_score,test_score,cv_score]
   
    print "neuron_count: "+str(neuron_count)+" "+str(cv_score)

plt.plot(scores_nn_df['neuron_count'],scores_nn_df['train_score'],label='train')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['test_score'],label='test')
plt.plot(scores_nn_df['neuron_count'],scores_nn_df['crossval_score'],label='crossval')
plt.xlabel('Neuron Count')
plt.ylabel('Score')
plt.title('Digits: Neural Network and KBest')
plt.legend()

plt.savefig('nn_kbest_learning_curve_digits.png')
plt.clf()