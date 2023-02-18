
import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler

ACTIONS = 9

class Player2:
    def __init__(self, alpha,beta,gamma,sigma_actor):
        
        self.sample_states = 100
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma 
        self.max_steps =45000
        self.sigma_actor = sigma_actor

        # sample states for the rbf
        state_space_samples = []
        env = gym.make('Pendulum-v1')
        for i_episode in range(100):
            observation = env.reset()
            for index in range(200):       
                x = observation
                state_space_samples.append(x)       
                action = env.action_space.sample()     
                observation, reward, done, info = env.step(action)
                if done:
                    break
        state_space_samples = np.array(state_space_samples)
        


        self.scaler = StandardScaler()
        self.scaler.fit(state_space_samples)
        self.featurizer= FeatureUnion([("rbf1", RBFSampler(gamma=4.0,n_components=self.sample_states)),
                                       ("rbf2", RBFSampler(gamma=1.0,n_components=self.sample_states))])
        self.featurizer.fit(self.scaler.transform(state_space_samples))
        # w weights for critic  
        self.w = np.zeros(( 2*self.sample_states,1))
        # theta weights for (mean) actor1
        self.t = np.zeros((2* self.sample_states,1))

   

    #---------------- X(S)--------------------------#
    # RBF mapping for a given state   
    def RBF(self,state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return np.array(featurized[0])
 
         
    # retrieve an action based on the policy distribution of a given state 
    def ret_action(self,state):
        state_rbf = self.RBF(state)
        #mu
        mu = np.dot((self.t).T,state_rbf)[0]
        #sigma
        sigma = self.sigma_actor       
        action = np.random.normal(mu, sigma, 1)
        return np.round( action[0],8)

    
    # gaussian policy score function (gradient) for theta
    def gradient_t(self,state,action): 
        state_rbf = self.RBF(state)
        mu = np.dot((self.t).T,state_rbf)[0]
        sigma = self.sigma_actor
        gradient_t = ((action - mu) * state_rbf)/(sigma**2)
        return np.round(gradient_t,8)
    
                                
    #---------------- <X(S,A),W>-----------------------#
    # approximate the state value action with the weights and the the feature vector
    def q_approximation(self,state):        
        state_rbf = self.RBF(state)
        approx = np.dot((self.w).T,state_rbf)[0]
        return approx

    def actor_critic(self):
        env = gym.make('Pendulum-v1')
        improvement_rate = {}
        steps_count = 0

        while steps_count < self.max_steps:
            start_state = env.reset()
            cur_state = start_state 
            action= env.action_space.sample()[0]                
            done = False
            episode_steps = 0
            
            while not done:                 
                steps_count +=1  
                episode_steps+=1
                new_state, reward, done, info = env.step([action])
                if done==True and episode_steps<200:
                    print("Achieved goal at step "+str(steps_count))
                    
                new_action = self.ret_action(new_state)  
                target = reward + self.gamma * self.q_approximation(new_state)   
                if reward >= -0.05 :
                    reward = 0
                    target = reward
                                        
                cur_q_approximation =  self.q_approximation(cur_state)
                cur_q_feature_vec = self.RBF(cur_state)   
                error =   target -  cur_q_approximation

                # update the theta and w weights with gradient descent
                self.t = self.t+ (self.alpha*error*self.gradient_t(cur_state,action)).reshape(2*self.sample_states,1)*(self.gamma**episode_steps)
                self.w = self.w + (self.beta*error*cur_q_feature_vec).reshape(2*self.sample_states,1)*(self.gamma**episode_steps)

                # update states
                cur_state = new_state
                action = new_action

                if steps_count % 1000 == 0:
                    print("number of steps "+str(steps_count))
                    improvement_rate[steps_count]=self.exp_rewards()
                  #  if steps_count % 10000 == 0:
                      #  self.exp_rewards_render() 
                      
            if steps_count >= self.max_steps:
                improvement_rate[steps_count]=self.exp_rewards()
                break                                
        return improvement_rate
        

    
    def exp_rewards(self):
        gamma = 1
        k=10
        rewards=0
        for sim in range(k):
            env = gym.make('Pendulum-v1')  	
            i = 0
            start_state = env.reset()
            action= env.action_space.sample()[0] 
            cur_state = start_state
            done = False
            while not done:
                new_state, reward, done, info = env.step([action])
                cur_state = new_state
                action = self.ret_action(cur_state)
                rewards+=reward*np.power(gamma,i)
                i += 1                  		   		
                if done:
                    break
        return rewards/k
    
     # used for rendering   
    def exp_rewards_render(self):
        gamma = 1
        k=5
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
                new_state, reward, done, info = env.step([action])
                cur_state = new_state
                action = self.ret_action(cur_state)
                rewards+=reward*np.power(gamma,i)
                i += 1                  		   		
                if done:
                    break
            env.close()
