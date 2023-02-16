
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)




def data_loader(dataset):
    
    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
        try:
            print("import MNIST from local directory")
            mnist_data = torchvision.datasets.MNIST('path/to/MNIST_root/', transform = transform)
        except:
            print("download MNIST to local directory")
            mnist_data = torchvision.datasets.MNIST('path/to/MNIST_root/',download = True, transform = transform)
            
            

        mnistX = mnist_data.data.float().to(device)
        mnisty = F.one_hot(mnist_data.targets).float().to(device)
        
        trainX = mnistX[:int(.6*len(mnistX))]
        valX = mnistX[int(.6*len(mnistX)):int(.8*len(mnistX))]
        testX = mnistX[int(.8*len(mnistX)):]
        
        trainy = mnisty[:int(.6*len(mnisty))]
        valy = mnisty[int(.6*len(mnisty)):int(.8*len(mnisty))]
        testy = mnisty[int(.8*len(mnisty)):]
        
        return trainX,trainy,valX,valy,testX,testy

    
    if dataset == 'Synthetic Data':
        data= np.random.uniform(0,1,(10000,50))
        x_i= np.random.randint(20,30,(10000))
        data[np.arange(len(data)), x_i+5] = data[np.arange(len(data)), x_i+5]*0.1
        data[np.arange(len(data)), x_i-5] = data[np.arange(len(data)), x_i-5]*0.1        
        train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
        train = torch.tensor(train).unsqueeze(2).float().to(device)
        val = torch.tensor(val).unsqueeze(2).float().to(device)
        test = torch.tensor(test).unsqueeze(2).float().to(device)    
        return train, val, test
        
datasets = ['MNIST','SP500','Synthetic Data'] 


#################### 3.1 ####################      

class AutoEncoder(nn.Module):
    def __init__(self,T ,n_features,n_hidden, num_layers =  2):
        super().__init__()
        #encoder
        self.lstm1 =  nn.LSTM(input_size = n_features
                             , hidden_size  = n_hidden
                             , num_layers = num_layers
                             , batch_first  = True)
        #decoder
        self.lstm2 =  nn.LSTM(input_size = n_hidden
                             , hidden_size  = n_hidden
                             , num_layers = num_layers
                             , batch_first  = True)
        
        self.lstm3 =  nn.LSTM(input_size = n_hidden
                             , hidden_size  = n_hidden
                             , num_layers = num_layers
                             , batch_first  = True)
        self.dropout = nn.Dropout(0.35)
        
        self.linear = nn.Linear(n_hidden, n_features)
        self.linear_clf = nn.Linear(n_hidden*T*num_layers , 10)

    def forward(self, x):

        seq_length = x.shape[1]
        
        # encoder
        out,(h_n,c_n) = self.lstm1(x)
        z = h_n.repeat(seq_length,1,1)
        z = torch.permute(z, (1, 0,2))
        
        #decode
        output,(h_n,_) = self.lstm2(z)
        clf = torch.flatten(output, 1)
        clf = self.dropout(clf)
        clf_output  = self.linear_clf(clf ) 
        x_tilde =self.linear(output)        
        return x_tilde,clf_output 
    
# plots input image   
def imshow(img):
    npimg = img.cpu().numpy()
    plt.imshow(npimg )
    plt.show()
 
# gets the label from the cross entropy output
def get_label(data):
    max_idx = torch.argmax(data, 1, keepdim=True).to(device)
    one_hot = torch.FloatTensor(data.shape).to(device)
    one_hot.zero_()
    output = one_hot.scatter_(1, max_idx, 1)
    return output

# NN trainer
def trainer_mnist(trainX, trainy, valX, valy, optimizer, lr, epochs, batch_size, hidden_size, clip_val, lambda1 = 1,lambda2 = 1):
    T = trainX.shape[1]
    d = trainX.shape[2]
    num_layers = 1
    net = AutoEncoder(T,d,hidden_size,num_layers).to(device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()

    val_loss = []
    train_loss = []
    val_loss_clf = []
    train_loss_clf = []
    train_acc = []
    val_acc =[]

    if optimizer =='SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optimizer =='adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)

    train_batches =  int(len(trainX)/batch_size +1)-1
    val_batches =  int(len(valX)/batch_size +1)-1    


    for epoch in range(epochs): 
        sum_acc = 0
        sum_loss_mse = 0
        sum_loss_softmax = 0

        for i in range(train_batches):
            
            batchX = trainX[i*batch_size:i*batch_size + batch_size]
            batchy = trainy[i*batch_size:i*batch_size + batch_size]
            
            optimizer.zero_grad()
            output = net(batchX)
           
            loss_mse = criterion1(output[0].float(), batchX.float())
            loss_softmax = criterion2(output[1], batchy)
            loss =   lambda1*loss_mse + lambda2*loss_softmax
                        
            sum_loss_mse += loss_mse.item()/train_batches 
            sum_loss_softmax += loss_softmax.item()/train_batches
            y_pred = get_label(output[1])

            #compute accuracy
            sum_acc += torch.sum((2*y_pred-batchy)==1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            optimizer.step()
        
        train_loss.append(sum_loss_mse)
        train_loss_clf.append(sum_loss_softmax)
        train_acc.append((sum_acc/(batch_size*train_batches)).cpu())       
            
        sum_acc = 0
        sum_loss_mse = 0
        sum_loss_softmax = 0
        for i in range(val_batches):
            
            batchX = valX[i*batch_size:i*batch_size + batch_size]
            batchy = valy[i*batch_size:i*batch_size + batch_size]

            with torch.no_grad():
                output = net(batchX)
            
            loss_mse = criterion1(output[0].float(), batchX.float())
            loss_softmax = criterion2(output[1], batchy)
            loss =   lambda1*loss_mse + lambda2*loss_softmax
            
            sum_loss_mse += loss_mse.item()/val_batches
            sum_loss_softmax += loss_softmax.item()/val_batches
            y_pred = get_label(output[1])
            sum_acc += torch.sum((2*y_pred-batchy)==1)
            
        val_loss.append(sum_loss_mse)
        val_loss_clf.append(sum_loss_softmax)
        val_acc.append((sum_acc/(batch_size*val_batches)).cpu()) 
        
        sum_acc = 0

        print("Epoch {}, loss {}, val loss {}, clf loss {},clf val loss {},train acc {},val acc {}  ".format(epoch,
                                                                                    train_loss[-1],val_loss[-1],
                                                                                    train_loss_clf[-1],val_loss_clf[-1],
                                                                                      train_acc[-1], val_acc[-1]))

    return net,train_loss, val_loss, train_loss_clf, val_loss_clf, train_acc, val_acc,train_acc,val_acc




trainX,trainy,valX,valy,testX,testy = data_loader('MNIST')
outputs = trainer_mnist(trainX, trainy, valX, valy, 'adam', 0.001, 30, 64,200, 1000, lambda2 = 0)
model = outputs[0]


#plot image reconstructions
image_num = 3
plot_set = testX[56:59]
for i in range(image_num):
    imshow(model(plot_set)[0].detach()[i])
    imshow(plot_set[i])
    
###################################### 3.2 ########################################################
trainX,trainy,valX,valy,testX,testy = data_loader('MNIST')
outputs = trainer_mnist(trainX, trainy, valX, valy, 'adam', 0.00025,60, 32,800, 100, lambda1 = 5/1000)
model = outputs[0]

imshow(model(testX[:100])[0].detach()[0])
imshow(testX[0])

plt.plot(outputs[1])
plt.plot(outputs[2])
plt.legend(['train','val'],  prop={'size': 15})
plt.ylabel('loss-reconstruction', size = 15)
plt.xlabel('epoch', size = 15)
plt.grid()
plt.show()


plt.plot(outputs[3])
plt.plot(outputs[4])
plt.legend(['train','val'],  prop={'size': 15})
plt.ylabel('loss-classify', size = 15)
plt.xlabel('epoch', size = 15)
plt.grid()
plt.show()


plt.plot(outputs[7])
plt.plot(outputs[8])
plt.legend(['train','val'],  prop={'size': 15})
plt.ylabel('accuracy-classify', size = 15)
plt.xlabel('epoch', size = 15)
plt.grid()
plt.show()

pred = model(testX)[1]
y_pred = get_label(pred)

torch.sum((2*y_pred-testy)==1)/len(testX)

###################################### 3.3 ########################################################
trainX,trainy,valX,valy,testX,testy = data_loader('MNIST')
trainX = trainX.flatten(1).unsqueeze(2)
valX = valX.flatten(1).unsqueeze(2)
testX = testX.flatten(1).unsqueeze(2)

outputs = trainer_mnist(trainX, trainy, valX, valy, 'adam', 0.001, 10, 16,30, 1000, lambda1 =1/1000)
model = outputs[0]
image_num = 3
plot_set = testX[56:59]
print(plot_set.shape)
for i in range(image_num):
    #print(model(plot_set)[0].shape)
    imshow(model(plot_set)[0].detach()[i].reshape(28,28,1))
    imshow(plot_set[i].reshape(28,28,1))

plt.plot(outputs[1])
plt.plot(outputs[2])
plt.legend(['train','val'],  prop={'size': 15})
plt.ylabel('loss-reconstruction', size = 15)
plt.xlabel('epoch', size = 15)
plt.grid()
plt.show()


plt.plot(outputs[3])
plt.plot(outputs[4])
plt.legend(['train','val'],  prop={'size': 15})
plt.ylabel('loss-classify', size = 15)
plt.xlabel('epoch', size = 15)
plt.grid()
plt.show()


plt.plot(outputs[7])
plt.plot(outputs[8])
plt.legend(['train','val'],  prop={'size': 15})
plt.ylabel('accuracy-classify', size = 15)
plt.xlabel('epoch', size = 15)
plt.grid()
plt.show()
