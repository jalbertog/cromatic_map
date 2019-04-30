import numpy as np
from numpy.linalg.linalg import pinv
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
 
 
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
    def __init__(self, input_n, hidden_shape, n_class):
        self.hidden_shape = hidden_shape
        self.hidden_neurons = np.zeros(shape=(0,input_n))
        self.spread = 0
        self.weights = 0
         
    def generateHiddenNeurons(self,X):
        kmeans = KMeans(n_clusters=self.hidden_shape).fit(X)
        self.hidden_neurons=np.array(kmeans.cluster_centers_)
        print 'labels ',kmeans.labels_
        return self.hidden_neurons
 
    def sigma(self):
        dTemp = 0
        for i in range(0,self.hidden_shape):
            for k in range(0,self.hidden_shape):
                dist = np.square(np.linalg.norm(self.hidden_neurons[i] - self.hidden_neurons[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp/np.sqrt(2*self.hidden_shape)
 
    def train(self,X,y):
        self.generateHiddenNeurons(X)
        self.sigma()
        hiddenOut = np.zeros(shape=(X.shape[0],self.hidden_shape))
        for i,item in enumerate(X):
            out=[]
            for j,proto in enumerate(self.hidden_neurons):
                distance = np.square(np.linalg.norm(item - proto))
                hiddenOut[i,j] = np.exp(-(distance)/(np.square(self.spread)))
                #out.append(neuronOut)
            #hiddenOut = np.vstack([hiddenOut,np.array(out)])
        #print hiddenOut
        #print ('hiddenOut',hiddenOut.shape)
        print(pinv(hiddenOut).shape,'X',y.shape)
        self.weights = np.dot(pinv(hiddenOut),y)
        print 'weights = ', self.weights.shape
 
    def test(self):
        items = [3,4,72,82,91]
        print('shape = ',self.scaledData[0].shape)
        print('weights = ',self.weights.shape)
        for item in items:
            data = self.scaledData[item]
            out = []
            for proto in self.hidden_neurons:
                distance = np.square(np.linalg.norm(data-proto))
                neuronOut = np.exp(-(distance)/np.square(self.spread))
                out.append(neuronOut)
            print ('out: ',out)
            netOut = np.dot(np.array(out),self.weights)
            print '---------------------------------'
            print netOut
            print 'Class is ',netOut.argmax(axis=0) + 1
            print 'Given Class ',self.labels[item].argmax(axis=0) +1
    
    def predict(self,X,y,probs = False):
        y_pred = []
        fails = 0.0
        for item in range(X.shape[0]):
            data = X[item]
            out = []
            for proto in self.hidden_neurons:
                distance = np.square(np.linalg.norm(data-proto))
                neuronOut = np.exp(-(distance)/np.square(self.spread))
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
                 
        
 
data = LoadAndScaleData('./data.csv')
scaledData, label = data.getRawData()
X_train,X_test, y_train, y_test = train_test_split(scaledData,label,test_size=0.35,shuffle=True)
print('train = ',X_train.shape,y_train.shape)
network = RBFNetwork(4,12,3)
network.train(X_train,y_train)
print ('predict======================')
y_predict = network.predict(X_test,y_test,False)
print (y_predict.shape)
label_y_t = []
label_pred = []
for i in range(y_test.shape[0]):
    label_y_t.append(y_test[i].argmax(axis=0) + 1)
    label_pred.append(y_predict[i].argmax(axis=0) + 1)
print (label_y_t,label_pred)
print cm(label_y_t,label_pred)
