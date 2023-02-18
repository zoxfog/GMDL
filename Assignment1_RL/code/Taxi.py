

import gym
import numpy as np
import random


class Taxi:
    def __init__(self,gama,n_env):
        # gama
        self.gama = gama
        # number of environments to learn
        self.n_env = n_env
        
        # initialize the following:
        # reward function, all rewards initialzed at -10
        self.reward_function = np.negative(np.ones((500 ,6 )  ))*10
        # transition function
        self.transition_function = np.negative(np.ones((500 ,6 ),dtype =  np.int32  ))
        # value function
        self.value_function = np.zeros((500))
        # policy function, with random policies
        self.policy_function = [] 
        for i in range(500):
            self.policy_function.append(random.choice([0,1,2,3,4,5]))
        # states that the learner has experienced
        self.known_states = []
        # end state that the learner has experienced
        self.end_states = []
        # start state that the learner has experienced
        self.start_states = []
        # policy return expectancy dictionary
        self.policy_exp = {}
        
    # function to update new state to an 'acknowledged' list of states
    def known_states_update(self,state):
        if state not in self.known_states:
            self.known_states.append(state)
            
     # function to update new state to an 'acknowledged' list of states
    def known_start_states_update(self,state):
        if state not in self.end_states:
            self.start_states.append(state)
            
     # function to update new state to an 'acknowledged' list of states
    def known_end_states_update(self,state):
        if state not in self.end_states:
            self.end_states.append(state)
            

    # get the expected reward of a given state, considering its sequence of policies (evaluate the policy)
    def exp_policy(self,state):
        
        total_reward = 0
        reward = 0
        current_state = state

        i = 0
        
        # limit an episode to 5000 steps
        while i<5000 and reward!=20:

            policy =self.policy_function[current_state]
            reward = self.reward_function[current_state,policy].copy()
            
            total_reward += round(reward*pow(self.gama, i),6)                
            next_state = self.transition_function[current_state,policy].copy()
            current_state = next_state 
            i += 1

        return total_reward
    

    # policy evaluation
    def policy_evaluation(self):
        delta = 1
        # loop until no values have changed
        while delta!=0:
            delta = 0
            for state in self.known_states:

                old_value = self.value_function[state].copy()                                                                              
                policy =self.policy_function[state]

                # get next state given action a
                next_state_a = self.transition_function[state,policy].copy()

                # if next state is known - proceed
                if next_state_a!=(-1):
                        val = 0
                        if state not in self.end_states:
                            reward = self.reward_function[state,policy].copy()
                            next_state_val = self.value_function[next_state_a].copy()
                            val += reward + next_state_val*(self.gama)
                            self.value_function[state] = val

                # get the delta between the value functions
                val_delta =  abs(self.value_function[state] - old_value) 
                val_delta = round(val_delta,6)
                
                # update delta
                if val_delta>=delta:
                    delta = val_delta

                    
    # policy improvement
    def policy_improvement(self):
        policy_stable = True
        for state in self.known_states:
            old_policies = self.policy_function[state]
            new_policies = old_policies
            state_q = {}
            
            # for every possible action, find the best q value -> best policy 
            for a in range(6):
                next_state_a = self.transition_function[state,a].copy()
                if next_state_a!=(-1):
                    
                    reward = self.reward_function[state,a].copy()
                    next_state_val = self.value_function[next_state_a].copy()
                    q = round(reward + next_state_val*(self.gama),6)
                    state_q[a] = q

                

            # get the q values of the policies with the best q values 
            if len(state_q)!=0:
                max_q =  state_q[max(state_q, key=state_q.get)]   
                max_state_q = {key:val for key, val in state_q.items() if val >=max_q}            
                new_policies = list(max_state_q.keys())[0]
           

            if new_policies!=old_policies:
                policy_stable = False
                self.policy_function[state] = new_policies
        return policy_stable


    def env_learn(self):
        print("Please wait, agent learning the states of "+ str(self.n_env)+" environments...")

        env = gym.make('Taxi-v3')

        # let the agent learn the evironments
        for i_episode in range(self.n_env):
            
            #reset environment
            env.reset()
            start_state = env.s
            self.known_states_update(start_state)
            self.known_start_states_update(start_state)
            current_state = start_state

            done = False           
            # while not finished learning all given states and actions
            while not done:
                
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)


                # make sure the environment does not terminate after 200 steps
                if done==True and reward!=20:
                    done= False
                
                # add terminal state to a list of terminal states
                elif reward==20:
                    self.known_end_states_update(next_state)
            
                # update transition and return functions  
                self.known_states_update(next_state) 
                self.transition_function[current_state,action] = next_state
                self.reward_function[current_state,action] = reward


                current_state = next_state

        iteration = 0

        # add expected return before policy iteration
        sum_exp_value = []
        for state in self.start_states:
            sum_exp_value.append(self.exp_policy(state))
            self.policy_exp[iteration] = np.mean(sum_exp_value)

        # policy iteration
        iteration = 0
        policy_stable = False

        # while imporvement takes place (not stable)
        while not policy_stable:

            # policy evaluation  
            self.policy_evaluation()
            # policy improvement
            policy_stable = self.policy_improvement()

            iteration +=1

            # add expected return at each iteration
            sum_exp_value = []
            for state in self.start_states:
                sum_exp_value.append(self.exp_policy(state))
            self.policy_exp[iteration] = np.mean(sum_exp_value)

        # close the evironment
        env.close()
        

        
