import numpy as np
from numpy.linalg.linalg import pinv
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix as cm, r2_score

 
class LoadAndScaleData:
    def __init__(self,filePath):
        self.data = np.loadtxt(filePath,delimiter=',',usecols=(0,1,2,3))
        self.labels = np.loadtxt(filePath,delimiter=',',dtype='S',usecols=(4))
        x, = self.labels.shape
        self.rLabels = np.zeros(shape = (0,3))
        for i in range(0,x):
            if self.labels[i] == 'Iris-setosa':
                self.rLabels = np.vstack([self.rLabels,[1,0,0]])
            if self.labels[i] == 'Iris-versicolor':
                self.rLabels = np.vstack([self.rLabels,[0,1,0]])
            if self.labels[i] == 'Iris-virginica':
                self.rLabels = np.vstack([self.rLabels,[0,0,1]])
     
    def getRawData(self):
        return np.asmatrix(self.data), self.rLabels          
    def scale(self):
        mat = np.asmatrix(self.data)
        height,width = mat.shape
        for i in range(0,width):
            minimum = np.min(mat[:,i])
            maximum = np.max(mat[:,i])
            for k in range(0,height):
                mat[k,i] = (mat[k,i] - minimum)/(maximum - minimum)
        return mat, self.rLabels
 
 
     
 
class RBFNetwork:
    def __init__(self, input_n, hidden_shape, n_class, cluster_by_class=False):
        self.hidden_shape = hidden_shape
        self.hidden_neurons = np.zeros(shape=(0,input_n))
        self.spread = 0
        self.weights = 0
        self.n_class = n_class
        self.activation = np.zeros(shape=(hidden_shape,))
        self.cluster_by_class = cluster_by_class
        self.accurracy = 0
         
    def generateHiddenNeurons(self,X, y=None):
        clusters = []
        if self.cluster_by_class and y is not None:
            #print '==========================Using clustering class================'
            classes = np.unique(y,axis=0)
            
            n_clusters = self.hidden_shape/int(classes.shape[0])
            
            self.hidden_shape = n_clusters*int(classes.shape[0])
            for n_class in classes:
                class_cluster = X[np.where(np.all(y == n_class,axis=1))]
                #print 'cluster = ',class_cluster.shape
                kmeans = KMeans(n_clusters=n_clusters).fit(class_cluster)
                self.hidden_neurons = np.vstack((self.hidden_neurons,np.array(kmeans.cluster_centers_)))
                #print 'hiddedns',self.hidden_neurons.shape
                local_clusters = []
                for i in range(kmeans.n_clusters):
                    clusters.append(class_cluster[np.where(kmeans.labels_ == i)])
#                local_clusters =  np.array([class_cluster[np.where(kmeans.labels_ == i)] for i in range(kmeans.n_clusters)])
                #print ('locals ',clusters)
                #clusters = np.vstack((clusters,local_clusters))
            clusters = np.array(clusters)    
            #print clusters.shape
        else:     
            #print '==============Using clustering gen============'
            kmeans = KMeans(n_clusters=self.hidden_shape).fit(X)
            self.hidden_neurons=np.array(kmeans.cluster_centers_)

            clusters = np.array([X[np.where(kmeans.labels_ == i)] for i in range(kmeans.n_clusters)])
        #print self.matrixSpread(clusters,self.hidden_neurons)
        return self.hidden_neurons
 
    def matrixSpread(self,clusters,centers):
        for i, (u, cluster) in enumerate(zip(centers,clusters)):
            cluster_dist = 0
            for vect in cluster:
                #print ('u = ',u, 'vect = ',vect[0])
                dist = np.square(np.linalg.norm(u - vect[0]))
                cluster_dist = np.max([cluster_dist,dist])
            self.activation[i] = cluster_dist
        return self.activation
            
    def maxSpread(self):
        dTemp = 0
        for i in range(0,self.hidden_shape):
            for k in range(0,self.hidden_shape):
                dist = np.square(np.linalg.norm(self.hidden_neurons[i] - self.hidden_neurons[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp/np.sqrt(2*self.hidden_shape)
 
    def train(self,X,y):
        self.generateHiddenNeurons(X,y)
        self.maxSpread()
        hiddenOut = np.zeros(shape=(X.shape[0],self.hidden_shape))
        for i,item in enumerate(X):
            out=[]
            for j,proto in enumerate(self.hidden_neurons):
                distance = np.square(np.linalg.norm(item - proto))
                spread = self.activation[j] if self.activation[j] - 0.0000001 > 0 else self.spread 
                #print 'spread ',spread
                hiddenOut[i,j] = np.exp(-(distance)/(np.square(spread)))
                #out.append(neuronOut)
            #hiddenOut = np.vstack([hiddenOut,np.array(out)])
        #print hiddenOut
        #print ('hiddenOut',hiddenOut.shape)
        print(pinv(hiddenOut).shape,'X',y.shape)
        self.weights = np.dot(pinv(hiddenOut),y)
        print 'weights = ', self.weights.shape
 
    
    def predict(self,X,y,probs = False):
        y_pred = []
        fails = 0.0
        for item in range(X.shape[0]):
            data = X[item]
            out = []
            for j,proto in enumerate(self.hidden_neurons):
                distance = np.square(np.linalg.norm(data-proto))
                spread = self.activation[j] if self.activation[j] - 0.0000001 > 0 else self.spread 
                neuronOut = np.exp(-(distance)/np.square(spread))
                out.append(neuronOut)
            netOut = np.dot(np.array(out),self.weights)
            if probs:
                y_pred.append(netOut)                
            else:
                out = np.zeros((3,),dtype=np.int)
                out[netOut.argmax(axis=0)] = 1;
                y_pred.append(out)
            
            print '---------------------------------'
            pred = netOut.argmax(axis=0) + 1
            given = y[item].argmax(axis=0) +1
            print 'Class is ',pred
            print 'Given Class ',given
            if pred != given:
                print 'FAIL!!'
                fails += 1
        print 'acurrate : ',1.0 - fails/(X.shape[0]*1.0)
        return np.array(y_pred)
    
    def score(self,X,y):
        y_pred = []
        fails = 0.0
        for item in range(X.shape[0]):
            data = X[item]
            out = []
            for j,proto in enumerate(self.hidden_neurons):
                distance = np.square(np.linalg.norm(data-proto))
                spread = self.activation[j] if self.activation[j] - 0.0000001 > 0 else self.spread 
                neuronOut = np.exp(-(distance)/np.square(spread))
                out.append(neuronOut)
            netOut = np.dot(np.array(out),self.weights)

            out = np.zeros((3,),dtype=np.int)
            out[netOut.argmax(axis=0)] = 1;
            y_pred.append(out)

            pred = netOut.argmax(axis=0) + 1
            given = y[item].argmax(axis=0) +1
            if pred != given:
                fails += 1
        self.accurracy = 1.0 - fails/(X.shape[0]*1.0)
        return self.accurracy      
                 
        
 
data = LoadAndScaleData('./data.csv')
scaledData, label = data.getRawData()
X_train,X_test, y_train, y_test = train_test_split(scaledData,label,test_size=0.5,shuffle=True)
print('train = ',X_train.shape,y_train.shape)
network = RBFNetwork(4,12,3,True)
network.train(X_train,y_train)
print ('predict======================')
y_predict = network.predict(X_test,y_test,False)
print (y_predict.shape)
label_y_t = []
label_pred = []
for i in range(y_test.shape[0]):
    label_y_t.append(y_test[i].argmax(axis=0) + 1)
    label_pred.append(y_predict[i].argmax(axis=0) + 1)
#print (label_y_t,label_pred)
print cm(label_y_t,label_pred)


################# KFOLD ################################

kfold = KFold(5, True, 10)
folds = []
for train, test in kfold.split(scaledData):
    net = RBFNetwork(4,12,3,True)
    net.train(scaledData[train],label[train])
    
    folds.append(net.score(scaledData[test],label[test]))
folds = np.array(folds)
print "scores",folds
print "kFold acurracy: ",folds.mean()


################## best k params #########################

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

kfold = KFold(5, True, 10)
test_train = list(kfold.split(scaledData))
prototypes = [9,12,18,25]

best = []
accuracy=np.zeros(len(prototypes))
for by_c in [True]:
    for i, prototype in enumerate(prototypes):

        folds = []
        for train, test in test_train:
            net = RBFNetwork(4,prototype,3,by_c)
            net.train(scaledData[train],label[train])
            
            folds.append(net.score(scaledData[test],label[test]))
        folds = np.array(folds)
        acc = folds.mean()
        accuracy[i] = acc
        best.append((prototype,acc,by_c))
print "best: ",best
plt.plot(prototypes,accuracy,'ro')
plt.ylabel('Accuracy %')
plt.xlabel('Beta')
plt.show()
plt.savefig('salida.jpg')
