import pandas as pd #used only for loading data
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

################### (1) ##################

SP500 =pd.read_csv("SP 500 Stock Prices 2014-2017.csv")

amazon = SP500[(SP500['symbol']=='AMZN')]
google= SP500[ (SP500['symbol']=='GOOGL')]
apple= SP500[ (SP500['symbol']=='AAPL')]

data = pd.DataFrame(index = amazon['date'].values)
data['GOOGL'] = google['high'].values
data['AMZN'] = amazon['high'].values
data['AAPL'] = apple['high'].values


fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (15,3))
ax[0].plot( pd.to_datetime(google['date']), google['high'].values )
ax[0].title.set_text('Google Daily Max Price')
ax[0].tick_params(labelrotation=45,axis='x')

ax[1].plot( pd.to_datetime(amazon['date']), amazon['high'].values )
ax[1].title.set_text('Amazon Daily Max Price')
ax[1].tick_params(labelrotation=45,axis='x')
plt.show()

################### (2) ####################

class AutoEncoder(nn.Module):
    def __init__(self,n_features,n_hidden, output_t = None, T = 50):
        super().__init__()
        
        if output_t==None:
            self.output_t = n_features
        #encoder
        self.lstm1 =  nn.LSTM(input_size = n_features
                             , hidden_size  = n_hidden
                             , num_layers = 1
                             , batch_first  = True)
        #decoder
        self.lstm2 =  nn.LSTM(input_size = n_hidden
                             , hidden_size  =  n_hidden
                             , num_layers = 1
                             , batch_first  = True)
        
        self.linear = nn.Linear(n_hidden, n_features)
        self.linear_forecast = nn.Linear(n_hidden,1)

    def forward(self, x):
        seq_length = x.shape[1]
        
        #encode
        out,(h_n,c_n) = self.lstm1(x)
        z = h_n.repeat(seq_length,1,1)
        z = torch.permute(z, (1, 0,2))
        
        #decode
        output,(h_n,c_n) = self.lstm2(z)
        reg_output = torch.flatten(torch.permute(h_n, (1, 0,2)), 1)

        x_tilde =self.linear(output)
        forecast_output = self.linear_forecast(reg_output)
        return x_tilde ,forecast_output
    
    
    
def preprocess(symbols, limit = 104, T = None, split = 'distinct', cut = 0, forecast = 1):
    if forecast is None:
        forecast = 0
    length = 1007 - cut
    X = []
    y = []
    scalers = []
    for s in symbols[:limit]:
        stock = SP500[(SP500['symbol']==s)].sort_values(by = 'date')
        stock_highs = stock['high'].values[:length]
        stock_highs = np.expand_dims(stock_highs, axis = 1)
        scaler = MinMaxScaler()
        stock_highs = scaler.fit_transform(stock_highs).squeeze(1)
        if len(stock_highs)>=length:
            scalers=scalers+[scaler]
        if split == 'distinct':
            if T is not None:
                v = np.array(np.array_split(stock_highs ,int(length/T)))
                stock_highs = np.vstack(np.array_split(stock_highs ,int(length/T)))
                scalers = scalers[:-1] + [MinMaxScaler()]*(int(length/T)-forecast)
            if len(X)==0:
                if len(symbols)==1:
                    X = stock_highs                        
                else:
                    X = stock_highs
            else:
                X=np.vstack((X,stock_highs))
        
        elif split == 'overlapping' and len(symbols)==1:
            m = T-forecast  # overlap size
            X = [stock_highs[i:i+T] for i in range(0, length-(forecast+T), 1)]
            y = [stock_highs[i:i+1] for i in range(T- 1 + forecast, length-1, 1)]


            X = np.array(X)
            y = np.array(y)
            
    X = torch.tensor(X).to(device)

    X = X.unsqueeze(2).float()
    y = torch.tensor(y).to(device)
    
    return X,y,scalers


def trainer_sp500(trainX, trainy, valX, valy, optimizer, lr, epochs, batch_size, hidden_size, clip_val, lambda1 = 1, lambda2 = 1):

    
    T = trainX.shape[1]
    d = trainX.shape[2]
    net = AutoEncoder(d,hidden_size, T=T).to(device)
    best_net = AutoEncoder(d,hidden_size, T=T).to(device)
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()

    val_loss = []
    train_loss = []
    val_loss_reg = []
    train_loss_reg = []
    train_mse = []
    val_mse =[]


    if optimizer =='SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optimizer =='adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)


    i_bathces =  int(len(trainX)/batch_size +1)
    best_loss = np.inf
    for epoch in range(epochs):  

        for i in range(i_bathces):
            
            batchX = trainX[i*batch_size:i*batch_size + batch_size]
            if trainy is  not None:
                batchy = trainy[i*batch_size:i*batch_size + batch_size]            
            optimizer.zero_grad()
            output = net(batchX)            
            loss_mse = criterion1(output[0].float(), batchX.float())
            
            if trainy is not None:
                loss_reg = criterion2(output[1].float(), batchy.float())
            else:
                loss_reg = 0
                
            loss = (loss_mse*lambda1  +  loss_reg*lambda2  )
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            optimizer.step()


        with torch.no_grad():
            val_output = net(valX)
            val_loss.append( criterion1(val_output[0], valX).item())
            if valy is not None:
                val_loss_reg.append( criterion2(val_output[1], valy).item())
            else:
                val_loss_reg.append(-1)
            val_mse.append(val_output[1])

            train_output = net(trainX)
            train_loss.append(criterion1(train_output[0], trainX).item())
            if trainy is not None:
                train_loss_reg.append( criterion2(train_output[1], trainy).item())
            else:
                train_loss_reg.append(-1)
            train_mse.append(train_output[1])
        print("Epoch {}, loss {}, val loss {}, reg loss {},reg val loss {},  ".format(epoch,
                                                                                    train_loss[-1],val_loss[-1],
                                                                                    train_loss_reg[-1],val_loss_reg[-1]))


    return net,train_loss, val_loss, train_loss_reg, val_loss_reg, train_mse, val_mse
            
T = 50
symbols_list = np.unique(SP500.symbol.values)
symbols_list = ['AMZN','AAPL','AAL','RSG','WMT','GOOGL','ROK', 'ROP','HOLX', 'HON', 'HP', 'HPE', 'HPQ']
symbols = pd.Series(np.unique(symbols_list),np.arange(len(np.unique(symbols_list))))
X,y,scalers  =  preprocess(symbols,T = 50, cut  = 507,split = 'distinct',forecast = 1)

trainX = X[:int(len(X)*0.6)]
valX = X[int(len(X)*0.6):int(len(X)*0.8)]
testX = X[int(len(X)*0.8):]

if len(y)!=0:
    trainy = y[:int(len(y)*0.6)]
    valy = y[int(len(y)*0.6):int(len(y)*0.8)]
    testy = y[int(len(y)*0.8):]
else:
    trainy = X[:int(len(X)*0.6)]
    valy = X[int(len(X)*0.6):int(len(y)*0.8)]
    testy = X[int(len(X)*0.8):]

outs1= []
for epochs in [5000]:
    out = trainer_sp500(trainX,None,valX,None, optimizer='adam', lr=0.0025, epochs =epochs, batch_size=32, hidden_size=100 , clip_val= 100)
    outs1.append(out)
    
    
#plot the reconstruction of HPQ and HPE

HPQ = testX[len(testX)-10:]
HPE = testX[len(testX)-20:len(testX)-10]

best_model = out[0]
output = best_model(HPE )[0]
output = output.cpu().detach().numpy()



fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (15,3))
ax[0].plot( pd.to_datetime(google['date'][:500]),HPE.cpu().ravel() )
ax[0].title.set_text('HPE')
ax[0].tick_params(labelrotation=45,axis='x')
ax[0].plot(pd.to_datetime(google['date'][:500]),output.ravel() )
ax[0].title.set_text('HPE')
ax[0].tick_params(labelrotation=45,axis='x')


output = best_model(HPQ )[0]
output = output.cpu().detach().numpy()
ax[1].plot( pd.to_datetime(google['date'][:500]),HPQ.cpu().ravel() )
ax[1].title.set_text('HPQ')
ax[1].tick_params(labelrotation=45,axis='x')
ax[1].plot(pd.to_datetime(google['date'][:500]),output.ravel() )
ax[1].title.set_text('HPQ')
ax[1].tick_params(labelrotation=45,axis='x')
plt.show()

######################## (3) ###################

T = 20
symbols_list = np.unique(SP500.symbol.values)
symbols_list = ['HP']
symbols = pd.Series(np.unique(symbols_list),np.arange(len(np.unique(symbols_list))))
X,y,scalers  =  preprocess(symbols,T = T, cut  = 7,split = 'overlapping',forecast = 1)


trainX = X[:int(len(X)*0.6)]
valX = X[int(len(X)*0.6):int(len(X)*0.8)]
testX = X[int(len(X)*0.8):]

if len(y)!=0:
    trainy = y[:int(len(y)*0.6)]
    valy = y[int(len(y)*0.6):int(len(y)*0.8)]
    testy = y[int(len(y)*0.8):]
else:
    trainy = X[:int(len(X)*0.6)]
    valy = X[int(len(X)*0.6):int(len(y)*0.8)]
    testy = X[int(len(X)*0.8):]

out = trainer_sp500(trainX, trainy, valX, valy, optimizer='adam', lr=0.0005, epochs =7500, batch_size=16, hidden_size=100 , clip_val= 1000, lambda1 = 1/100)
out3 = out

plt.plot(out3[3])
plt.plot(out3[4])
plt.title("Prediction - loss vs epochs")
plt.legend(['train','val'])
plt.show() 

plt.plot(out3[1])
plt.plot(out3[2])
plt.title("Reconstruction - loss vs epochs")
plt.legend(['train','val'])
plt.show() 

model = out3[0]
#temp_test = trainX[0].unsqueeze(0)
datasets  = [(trainX,trainy),(valX,valy),(testX,testy) ]
for dataset in datasets:
    predictions = []
    X,y = dataset
    for series_x in X:
        with torch.no_grad():
            temp_s = series_x.unsqueeze(0)
            pred = model(temp_s)[1]
            predictions.append(pred[0][0].item())

    predictions = scalers[0].inverse_transform(np.expand_dims(predictions,1))
    true = scalers[0].inverse_transform(y.cpu().detach().numpy())

    plt.plot(true)
    plt.plot(predictions)
    plt.legend(['true values','prediction'])
    plt.show() 
    
######################### (4)   ######################
model = out3[0]

datasets  = [(trainX,trainy),(valX,valy),(testX,testy) ]
for dataset in datasets:
    predictions = []
    X,y = dataset
    multistep_X = X[:,:int(T/2)]
    for series_x in X:
        temp_s = series_x.unsqueeze(0)
        for i in range(int(T/2)+1):
            with torch.no_grad():
                pred = model(temp_s)[1]
            if i==int(T/2):    
                predictions.append(pred[0][0].item())
            pred = pred.unsqueeze(0)
            temp_s = temp_s.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            
            temp_s = np.concatenate([temp_s,pred],axis =1)
            temp_s  = temp_s[:,1:]
            temp_s = torch.tensor(temp_s).to(device)
    predictions = scalers[0].inverse_transform(np.expand_dims(predictions,1))
    true = scalers[0].inverse_transform(y.cpu().detach().numpy())

    plt.plot(true)
    plt.plot(predictions)
    plt.legend(['true values','prediction'])
    plt.show() 
