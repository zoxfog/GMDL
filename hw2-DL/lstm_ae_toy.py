import pandas as pd #used only for loading data
import numpy as np
import matplotlib.pyplot as plt
import itertools


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# This function loads the required datasets of MNIST or the Synthetic Data
def data_loader(dataset):
    if dataset == 'MNIST':
        try:
            print("import MNIST from local directory")
            mnist_data = torchvision.datasets.MNIST('path/to/MNIST_root/')
        except:
            print("download MNIST to local directory")
            mnist_data = torchvision.datasets.MNIST('path/to/MNIST_root/',download = True)

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
        for row in data:
            x_i = np.random.randint(20,30)
            row[x_i-5:x_i+5] = row[x_i-5:x_i+5]*0.1  
        train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
        train = torch.tensor(train).unsqueeze(2).float().to(device)
        val = torch.tensor(val).unsqueeze(2).float().to(device)
        test = torch.tensor(test).unsqueeze(2).float().to(device)    
        return train, val, test
        
datasets = ['MNIST','SP500','Synthetic Data']   
        
 

# plot examples of the synthetic data
s_data,_,_ = data_loader(datasets[2])
rand = np.random.randint(0,len(s_data))
plt.figure(figsize=(19,4))
plt.plot(s_data[rand].cpu())
plt.ylabel('Synthetic signal values',size = 20)
plt.xlabel('Timestep',size = 20)
plt.grid()
plt.show()  



class AutoEncoder(nn.Module):
    def __init__(self,n_features,n_hidden ):
        super().__init__()
    
        # encoder
        self.lstm1 =  nn.LSTM(input_size = n_features
                             , hidden_size  = n_hidden
                             , num_layers = 1
                             , batch_first  = True)
        # decoder
        self.lstm2 =  nn.LSTM(input_size = n_hidden
                             , hidden_size  = n_hidden
                             , num_layers = 1
                             , batch_first  = True)
        
        # output layer
        self.linear = nn.Linear(n_hidden, n_features)


    # input x shape: ( batch,T length, input size: d for multivariate or 1 for univarriate  )
    def forward(self, x):

        seq_length = x.shape[1]

        #encode
        _,(h_n,c_n) = self.lstm1(x)
        z = h_n.repeat(seq_length,1,1)
        z = torch.permute(z, (1, 0,2))

        #decode
        output,_ = self.lstm2(z)
        x_tilde =self.linear(output)        
        return x_tilde
 
# trainer function for the nn
def trainer_toy(train, val, optimizer, lr, epochs, batch_size, hidden_size, clip_val):

    d = 1 # input d=1, (univariate)
    net = AutoEncoder(d,hidden_size).to(device)
    criterion = nn.MSELoss()

    val_loss = []
    train_loss = []

    if optimizer =='SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optimizer =='adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)

    train_batches = torch.tensor_split(train, int(len(train)/batch_size +1) )

    for epoch in range(epochs):  
        for batch in train_batches:
            optimizer.zero_grad()
            output = net(batch)

            loss = criterion(output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            optimizer.step()

        with torch.no_grad():
            val_output = net(val)
            val_loss.append( criterion(val_output, val).item())

            train_output = net(train)
            train_loss.append(criterion(train_output, train).item())
        print("Epoch {}, loss {}, val loss {}".format(epoch,train_loss[-1],val_loss[-1]))
        
    return net,train_loss,val_loss


# hyperparameter tuning
dataset = 'Synthetic Data'

optimizer_list = ['adam','SGD']
lr_list = [0.001,0.01]
epochs_list = [200]
batch_list = [32]
hidden_list = [20,30,40]
clip_list= [1,10]


train, val, test = data_loader(dataset)
col = ['optimizer', 'lr', 'epochs', 'batch_size', 'hidden_size' , 'clip_val', 'val loss']
hyper_params = pd.DataFrame(columns = col)

for params in itertools.product(optimizer_list, lr_list, epochs_list, batch_list, hidden_list,clip_list):
    optimizer, lr, epochs, batch_size, hidden_size , clip_val = params 
    print("$$$$ -- optimizer {}, lr {}, epochs {}, batch {}, hidden {},clip {}  -- $$$$".format( optimizer, lr, epochs, batch_size, hidden_size , clip_val))
    net,train_loss,val_loss = trainer_toy(train ,val, optimizer, lr, epochs, batch_size, hidden_size , clip_val)
    hyper_params.loc[len(hyper_params)] = [optimizer, lr, epochs, batch_size, hidden_size , clip_val, val_loss[-1]]

# get the best network with the best params
hyper_params = hyper_params.sort_values(by = "val loss")
optimizer, lr, epochs, batch_size, hidden_size, clip_val,_ = hyper_params.iloc[0].values

train, val, test = data_loader(dataset)
net,train_loss,val_loss = trainer_toy(train ,val, optimizer, lr, epochs, batch_size, hidden_size , clip_val)
best_model = net

# plot the reconstruction
orig = test[:2]
with torch.no_grad():
    rec = best_model(orig ).cpu().detach().numpy()


fig, ax = plt.subplots(nrows=1, ncols=2,figsize = (20,3))


ax[0].plot(orig[0].cpu() )
ax[0].plot(rec[0]  )
ax[0].legend(["original","reconstruction"],loc="upper right")
ax[1].plot(orig[1].cpu())
ax[1].plot(rec[1]   )
ax[1].legend(["original","reconstruction"],loc="upper right")
plt.show()