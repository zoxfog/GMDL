



import time
import gym
import numpy as np
import imageio
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.autograd as autograd
import random
import pandas as pd
import os
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

print(torch.__version__ )
import gym
print(gym.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
print(torch.cuda.get_device_name(0))


# ## NN Impelementation




# Class for the classic DQN
class DQN(nn.Module):
    def __init__(self, h, w, actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)        
        # fully connected layers
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, actions)

        
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





# Class for the Dueling DQN
class Dueling_DQN(nn.Module):
    def __init__(self,h, w,  actions):
        super(Dueling_DQN, self).__init__()
        self.actions = actions        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1_adv = nn.Linear(7*7*64, 512)
        self.fc1_val = nn.Linear(7*7*64, 512)
        self.fc2_adv = nn.Linear(512, self.actions)
        self.fc2_val = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.actions)        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.actions)
        return x


# ## Screen Preprocessing



# transform the image to 3x84x84 from 3x250x160
def transform(screen):
    processed_image = cv2.resize(screen, (84, 84))
    processed_tensor = torch.tensor(processed_image.astype('uint8')/255).permute(2, 0 ,1)
    return processed_tensor.unsqueeze(0)


# ## Replay Buffer



#Replay Buffer Class

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    
    def __init__(self, capacity, history_len = 3 ):
        self.buffer = []
        self.capacity = capacity 
        self.history_len = 4

    def push(self, *args):
        self.buffer.append(Transition(*args))
        if len(self.buffer) == self.capacity:
            self.buffer = self.buffer[1:]
                                   
    def sample(self, batch_size):
        buffer_size = len(self.buffer)-1
        output = []
        tr = batch_size
        
        while tr>0:
            i = random.randint(0, buffer_size)
            output.append(self.buffer[i])
            tr -=1
            # get the history transitions
            j = i
            while j>0 and (i-j)<self.history_len and tr>0:
                j-=1
                output.append(self.buffer[j])
                tr -=1                        
        return output

    def __len__(self):
        return len(self.buffer)


# ## E-greedy Action Generator




def ret_action( state, epsilon,main_net, actions):
    r = random.random()
    if r > epsilon:
        with torch.no_grad():
            output = main_net(state).cpu()[0]
            greedy_actions = np.where(output.numpy()==np.max(output.numpy() ))[0]
            greedy_action =  random.choice(greedy_actions )          
        action = greedy_action
    else:
        action = random.randint(0, actions - 1)
    return action





def clipped_MSE(output, target):
    loss = torch.mean(torch.clamp((output - target), min=-1, max=1,)**2)
    return loss


# ## NN update for each batch

# In[19]:


def update_model(main_net,target_net, memory, batch_size, gamma):
    if len(memory) <  batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    Q_s_a = main_net(state_batch).gather(1, action_batch).double()
    next_state_values = torch.zeros( batch_size, device=device).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    

    
    # Compute the expected Q values
    next_Q_s_a = (next_state_values * gamma) + reward_batch
            

    # Compute the loss
    if type(main_net)==type(DQN(0,0,0)):
        loss= clipped_MSE(Q_s_a, next_Q_s_a) 
    if type(main_net)==type(Dueling_DQN(0,0,0)):
        loss =  nn.MSELoss()(Q_s_a, next_Q_s_a)

    return loss


# ## Simulator




def simulation(main_net,game):
    HEIGHT = 84
    WIDTH = 84

    GAME = game
    env = gym.make(GAME)    
    s = str(env.action_space)
    val = s.split('(', 1)[1].split(')')[0]
    ACTIONS= int(val) 
    
    
    episodes = 5
    exp_rewards_mean =[]
    exp_rewards_top_std =[]
    exp_rewards_bottom_std =[]
    epsilon = 0.05
    k = 1
    e_rewards = []

    #initilize
    for e in range(episodes):
        total_rewards = 0
        total_steps = 0
        step  = 0
        screen = env.reset()
        tr_current_screen = transform(screen )
        done = False
        # while not terminal
        while done==False and step<5000:
            
            if step%k==0 :
                action = ret_action(tr_current_screen,epsilon,main_net,ACTIONS)                 
            if step%50==0:
                action = ret_action(tr_current_screen,1,main_net,ACTIONS) 
                
            last_screen, reward, done, info = env.step(action)
            total_rewards+=reward   
            
            tr_last_screen= transform(last_screen)

            next_state = tr_current_screen - tr_last_screen
            
            # update state
            tr_current_screen = tr_last_screen
            total_steps = total_steps + 1
            step = step + 1        
        e_rewards.append(total_rewards)
    mean = np.mean(e_rewards)
    std_top = np.mean(e_rewards) + np.std(e_rewards)
    std_bottm = np.mean(e_rewards) - np.std(e_rewards)
    
    return mean, std_top, std_bottm


# ## Learner



# This function get all the hyper parameters as input and learn the environemnt with a given model
def learn_new(directory, model, max_episodes, max_steps , game = 'Breakout-v0' ,
              gamma = 0.99, batch_size = 32, optimizer = 'RMSprop', lr=0.00025, momentum = 0.95,
              eps = 0.01,epsilon_decay = 0.98, buffer_capacity = 100000, history_length = 4 ):
       
    # The directory for the checkpoitns
    DIRECTORY_NAME = directory
    
    # hyperparameters
    HEIGHT = 84
    WIDTH = 84
    GAME = game
    MAX_EPISODES = max_episodes
    MAX_STEPS = max_steps
    MODEL = model
    
    env = gym.make(GAME)
    s = str(env.action_space)
    val = s.split('(', 1)[1].split(')')[0]
    ACTIONS= int(val)

    GAMMA = gamma
    BATCH_SIZE = batch_size
    TARGET_UPDATE = 10000
    
    MAX_EPSILON = 0.9
    MIN_EPSILON = 0.1
    EPSILON_DECAY = epsilon_decay
    CAPACITY = buffer_capacity
       
    OPTIMIZER = optimizer
    LEARNING_RATE = lr
    MOMENTUM = momentum
    WEIGHT_DECAY = 0
    EPS = eps

    if MODEL == 'DQN':
        memory = ReplayBuffer(CAPACITY)
        main_net = DQN(HEIGHT, WIDTH, ACTIONS).double().to(device)
        target_net = DQN(HEIGHT, WIDTH, ACTIONS).double().to(device)
        target_net.eval()
    if MODEL == 'Dueling':
        memory = ReplayBuffer(CAPACITY)
        main_net = Dueling_DQN(HEIGHT, WIDTH, ACTIONS).double().to(device)
        target_net = Dueling_DQN(HEIGHT, WIDTH, ACTIONS).double().to(device)
        target_net.eval()

    if OPTIMIZER ==  "RMSprop":
        optimizer = optim.RMSprop(main_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, eps =  EPS,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER ==  "SGD":
        optimizer = optim.SGD(main_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, eps =  EPS,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER ==  "ADAM":
        optimizer = optim.Adam(main_net.parameters(), lr=LEARNING_RATE, eps =  EPS, weight_decay=WEIGHT_DECAY)
    
    # hyperparameter dictionary
    hyperparameter_dict = {"model": MODEL,"gamma":GAMMA,"batch size":BATCH_SIZE, "target update": TARGET_UPDATE,
                       "max epsilon": MAX_EPSILON, "min epsilon": MIN_EPSILON, "epsilon decay": EPSILON_DECAY,"game":GAME,
                          "max steps":MAX_STEPS, "max episodes": MAX_EPISODES}

    # Replay buffer hyperparameters hyperparameter dictionary
    rb_hyperparameter_dict ={"capacity":CAPACITY }

    opt_hyperparameters = {"optimizer": OPTIMIZER, 'learning rate': LEARNING_RATE,"momentum": MOMENTUM,
                         "eps": EPS , "weight decay":WEIGHT_DECAY}
    hyperparameters = {**hyperparameter_dict, **rb_hyperparameter_dict, **opt_hyperparameters}
    
   # create a directory to save the neural network's checkpoint parameters and checkpoint info   
    path = os.path.abspath(os.getcwd())
    dir_path = os.path.join(path, DIRECTORY_NAME)    
    try:
        os.mkdir(dir_path)
    except:
        print("Directory already exists")
    check_points = pd.DataFrame()

    # initiliaze
    total_steps = 0
    epsilon = MAX_EPSILON
    episode_num = 0
    training_loss = np.array([])            
    k = 4     #skipped frames
    noop = 30    # noop actions
    
    # expected reward 
    exp_rewards = []

    # initliaze time managment variables
    state_transformer_time = 0
    env_step_time = 0
    buffer_push_time = 0
    net_update_time = 0
    target_net_log_time = 0 
    total_time = 0

    while total_steps<MAX_STEPS and episode_num<MAX_EPISODES:
        episode_num+=1
        screen = env.reset()
        tr_current_screen = transform(screen )        
        total_rewards = 0
        step  = 0
        done =False

        # while not terminal
        while done==False:
            
            start_time = time.time()  
                 
            # take action in the current env
            if step%k==0:
                if step>noop and total_steps>30000:
                    action = ret_action(tr_current_screen, epsilon,main_net, ACTIONS) 
                elif step>noop:
                    action =  ret_action(tr_current_screen, 1,main_net, ACTIONS) 
                else:
                    action = 0

            next_screen, reward, done, info = env.step(action)
            total_rewards+=reward 
            
            # balance the rewards
            if reward>=1:
                reward = 1.0
            time_checkpoint1 = time.time()
            

            if not done:
                # transform the state to a 3x84x84 grey scale image   
                tr_next_screen= transform(next_screen)
            else:
                tr_next_screen = None
            time_checkpoint2 = time.time()

            # add transition to the buffer
            memory.push(tr_current_screen, torch.tensor([[action]]),
                        tr_next_screen, torch.tensor([reward]))
            time_checkpoint3 = time.time()

            # compute the loss and run a single update
            loss = update_model(main_net,target_net, memory, BATCH_SIZE, GAMMA )
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                for param in main_net.parameters():
                    param.grad.data.clamp_(max = 10)
                optimizer.step()           
                training_loss = np.append(training_loss, loss.item())
            time_checkpoint4 = time.time()

            # update state
            tr_current_screen = tr_next_screen
            
            
            total_steps = total_steps + 1
            step = step + 1        
            time_checkpoint5 = time.time()

            # update the target weights 
            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(main_net.state_dict())
            time_checkpoint6 = time.time()

            # update epsilon
            if total_steps%10000==0:
                if epsilon> MIN_EPSILON:
                    epsilon*=EPSILON_DECAY
            time_checkpoint7 = time.time()

            # add elapsed time in minutes
            env_step_time += (time_checkpoint1 - start_time)/60
            state_transformer_time += (time_checkpoint2 - time_checkpoint1)/60
            buffer_push_time += (time_checkpoint3-time_checkpoint2)/60
            net_update_time += (time_checkpoint4-time_checkpoint3)/60
            target_net_log_time += (time_checkpoint6-time_checkpoint5)/60 
            total_time += (time_checkpoint7 - start_time)/60        

            if done:
                print("Episode "+ str(episode_num)+", timestep "+str(total_steps) +
                      ", finished after {} timesteps".format(step+1))

                # checkpoint: save model and info
                checkpoint_episode = 2
                if episode_num%checkpoint_episode==0 and episode_num>0:
                    avg_reward = np.mean(exp_rewards)
                    std_reward = np.std(exp_rewards)
                    exp_rewards = []

                    nn_name = './main_net'+str(episode_num )+'.pth'
                    training_loss_name = '//training loss' +str(episode_num )+ '.npy'                                              

                    check_point = {"episode num":episode_num  ,"step":total_steps ,"epsilon": epsilon,
                                   "avg reward":avg_reward,"std reward":std_reward,"path":DIRECTORY_NAME+nn_name,
                                  "training loss path": DIRECTORY_NAME+training_loss_name}

                    avg_elapsed_time = {'state transformer time': state_transformer_time/checkpoint_episode,
                                        'env step time': env_step_time/checkpoint_episode,
                                        'buffer push time': buffer_push_time/checkpoint_episode,
                                        'net update time': net_update_time /checkpoint_episode,
                                        'target net update time': target_net_log_time/checkpoint_episode,
                                        'avg time': total_time/checkpoint_episode}

                    mean,top_std,bottom_std = simulation(main_net,GAME)
                    scores = {'mean':mean,'top std':top_std,'bottom_std':bottom_std}

                    # join the checkpoint dictionary and the hyperparameters data
                    check_point ={**check_point, **hyperparameters,**avg_elapsed_time,**scores}
                    check_points = check_points.append(check_point,ignore_index=True)
                    check_points.to_excel(DIRECTORY_NAME + "//checkpoints.xlsx",index = False) 
                    torch.save(main_net.state_dict(), DIRECTORY_NAME+nn_name)
                    np.save(DIRECTORY_NAME + training_loss_name,training_loss)

                    state_transformer_time = 0
                    env_step_time = 0
                    buffer_push_time = 0
                    net_update_time = 0
                    target_net_log_time = 0 
                    total_time = 0
                break
        exp_rewards.append(total_rewards) 
    


# ## Run Block




# learn the atari game

directory = 'dir11_Dueling'
model = 'Dueling'
max_episodes = 10000
max_steps = 5000000
game = 'Breakout-v0'
gamma = 0.99
batch_size =32
optimizer = 'ADAM'
lr = 0.00025
momentum = 0.95
eps = 0.01
epsilon_decay = 0.99
buffer_capacity = 100000
history_length = 4

learn_new(directory, model, max_episodes, max_steps , game  ,
              gamma , batch_size , optimizer , lr, momentum ,
              eps ,epsilon_decay, buffer_capacity , history_length)





