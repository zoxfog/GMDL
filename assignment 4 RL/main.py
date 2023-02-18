
from player2 import Player2
import matplotlib.pyplot as plt
import numpy as np
import gym


gamma = 0.99
alpha = 0.005
beta = 0.1
sigma_actor = 1.1

# run example
print("####### actor-critic ########")
print( "please wait a couple of minutes, applying actor-critic for 45,000 iterations")

player = Player2(alpha,beta,gamma,sigma_actor)
performance_qlearning = player.actor_critic()


#%%
keys = np.round(np.fromiter(performance_qlearning.keys(), dtype=float),4)/1000
vals = np.round(np.fromiter(performance_qlearning.values(), dtype=float),4)
plt.plot(keys,vals)
plt.title("actor critic"+'\n'+ "alpha="+str(alpha)+", beta="+str(beta)+", gamma="+str(1))
plt.xlabel('1000 steps', fontsize=11)
plt.show()

def simulation(player):
    gamma = 1
    k=1
    rewards=0
    for sim in range(k):
        env = gym.make('Pendulum-v1')  	
        i = 0
        start_state = env.reset()
        action= env.action_space.sample()[0] 
        cur_state = start_state
        done = False
        while not done:
            env.render()
            old_state = env.state
            new_state, reward, done, info = env.step([action])
            print(str(i+1)+". action: "+str(np.round(action,3))
                  +"state: theta"+str(np.round(old_state[0],3))
                  +", theta dot:"+str(np.round(old_state[1],3))
                  +", next state: "+str(np.round(env.state[0]))
                  +", theta dot:"+str(np.round(env.state[1]))
                  +", reward: "+str(np.round(reward,3)))
            cur_state = new_state
            action = player.ret_action(cur_state)
            rewards+=reward*np.power(gamma,i)
            print(reward)
            i += 1  
             		   		
            if done:
                break
        
        env.close()
        print("Sum Rewards: " +str(rewards) )
        
#%%

simulation(player)
