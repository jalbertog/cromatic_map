import numpy as np
from numpy.linalg.linalg import pinv
 
 
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
    def __init__(self, pTypes,scaledData,labels):
        self.pTypes = pTypes
        self.protos = np.zeros(shape=(0,4))
        self.scaledData = scaledData
        self.spread = 0
        self.labels = labels
        self.weights = 0
         
    def generatePrototypes(self):
        group1 = np.random.randint(0,49,size=self.pTypes)
        group2 = np.random.randint(50,100,size=self.pTypes)
        group3 = np.random.randint(101,150,size=self.pTypes)
        self.protos = np.vstack([self.protos,self.scaledData[group1,:],self.scaledData[group2,:],self.scaledData[group3,:]])
        print('protos.shape',self.protos.shape)
        return self.protos
 
    def sigma(self):
        dTemp = 0
        for i in range(0,self.pTypes*3):
            for k in range(0,self.pTypes*3):
                dist = np.square(np.linalg.norm(self.protos[i] - self.protos[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp/np.sqrt(self.pTypes*3)
 
    def train(self):
        self.generatePrototypes()
        self.sigma()
        hiddenOut = np.zeros(shape=(0,self.pTypes*3))
        print ('protos = ', self.protos[0])
        for item in self.scaledData:
            out=[]
            for proto in self.protos:
                distance = np.square(np.linalg.norm(item - proto))
                neuronOut = np.exp(-(distance)/(np.square(self.spread)))
                out.append(neuronOut)
            hiddenOut = np.vstack([hiddenOut,np.array(out)])
        #print hiddenOut

        self.weights = np.dot(pinv(hiddenOut),self.labels)
        print 'weights = ', self.weights
 
    def test(self):
        items = [3,4,72,82,91,120,134,98,67,145,131]
        print('shape = ',self.scaledData[0].shape)
        print('weights = ',self.weights.shape)
        for item in items:
            data = self.scaledData[item]
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data-proto))
                neuronOut = np.exp(-(distance)/np.square(self.spread))
                out.append(neuronOut)
            print ('out: ',out)
            netOut = np.dot(np.array(out),self.weights)
            print '---------------------------------'
            print netOut
            print 'Class is ',netOut.argmax(axis=0) + 1
            print 'Given Class ',self.labels[item].argmax(axis=0) +1
         
                 
        
 
data = LoadAndScaleData('./data.csv')
scaledData, label = data.scale()
network = RBFNetwork(4,scaledData,label)
network.train()
network.test()
