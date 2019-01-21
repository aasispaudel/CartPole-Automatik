
# coding: utf-8

# ## **DQN**

# In[ ]:


from collections import deque
from keras.layers import Dense, Conv1D, Reshape, Flatten
from keras.models import Sequential
import numpy as np
import random
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=3000)
        self.learning_rate = 0.001
        
        self.model = self._build_model_convolution()
        
        # If you want to work with prebuilt model comment the above line and
        # uncomment below
        # from keras.models import load_model
        # self.model = load_model(REPLACE THIS WITH DIR TO YOUR MODEL)
        # self.epsilon = self.epsilon_min
        
    def _build_model_convolution(self):
        model = Sequential()
        print(self.state_size)
        
        model.add(Reshape((self.state_size, 1)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
      
      
#    ###========================================================================####
#        If you want to build dense model
#    ###========================================================================####

#     def _build_model_dense(self):
#       model = Sequential()
#       model.add(Dense(64, activation='relu'))
#       model.add(Dense(32, activation='relu'))
#       model.add(Dense(self.action_size, activation='linear'))
#       model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
#       return model
    
#    ###========================================================================####
    
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        
    def act(self, state):
        if self.epsilon >= np.random.rand():
            return random.randrange(self.action_size)
        
        return np.argmax(self.model.predict(state)[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
              target[0][action] = reward
              
            else:
              Q_future = np.amax(self.model.predict(next_state)[0])
              target[0][action] = reward + self.gamma * Q_future
              
            self.model.fit(state, target, epochs=1, verbose=0)
            
        if self.epsilon_min < self.epsilon:
            self.epsilon *= self.epsilon_decay
            
    
  
#    ###======================================================================###
#        If you want to train with all the data you have collection
#        Warining: This might be considerably slow training, so better to train 
#                  from with each episode (remember while calling the function)
#    ###======================================================================###

#     def replay_with_all(self):
#         np.random.shuffle(self.memory)
#         for state, action, reward, next_state, done in self.memory:
#             target = self.model.predict(state)
#             if done:
#               target[0][action] = reward
#             else:
#               Q_future = np.amax(self.model.predict(next_state)[0])
#               target[0][action] = reward + self.gamma * Q_future
#             self.model.fit(state, target, epochs=1, verbose=0, batch_size=64)
#         if self.epsilon_min < self.epsilon:
#             self.epsilon *= self.epsilon_decay

#     ###===============================================================###    

    def save_model(fn):
       self.model.save(fn)


# # Training **Agent** 

# In[ ]:


import gym

def main():
    no_episodes = 2000
    batch_size = 32
    
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(env)
    for e in range(no_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        for t in range(500):
            # If you want to render
            # env.render()
            
            action = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -10
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                print(f'Episode: {e}   Score: {t}   Epsilon: {agent.epsilon}')
                
                if t >= 199:
                  print("\nSUCCESS")
                  agent.save_model('./catpole_player/cp.model')
                  return
                  
                break
            
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # agent.replay_with_all()


# In[ ]:


if __name__ == '__main__':
    main()

