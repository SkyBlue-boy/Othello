import numpy as np
import tensorflow as tf
import tensorflow.keras as tfs
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import hashlib

class DQN:
    def __init__(self, state_shape, action_size, replay_buffer_size=10000, batch_size=128, gamma=0.99, lr=0.001):
        self.action_size = action_size
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.gamma = gamma
        self.lr = lr
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.hashtable = [[{} for _ in range(15)] for _ in range(15)] 
        self.maxBufferSize = replay_buffer_size
        self.replay_buffer = []

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)

    def _build_model(self):
        state_input = layers.Input(shape=self.state_shape)
        turn_input = layers.Input(shape=(1,), dtype='float32')
        # turn_input을 3차원 텐서로 변환
        turn_info = layers.RepeatVector(self.state_shape[0] * self.state_shape[1])(turn_input)
        turn_info = layers.Reshape((self.state_shape[0], self.state_shape[1], 1))(turn_info)

        # 상태 텐서와 턴 정보를 결합
        combined_input = layers.Concatenate(axis=-1)([state_input, turn_info])
        x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(combined_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # Residual block 1
        for _ in range(3):
            shortcut = x
            x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)
        
        # Residual block 2
        for i in range(4):
            shortcut = x
            if i == 0:  # 첫 번째 블록에서만 채널 수가 변경되므로, 이를 맞추기 위해 1x1 Convolution 사용
                shortcut = layers.Conv2D(128, kernel_size=1, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)

        # Global average pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = models.Model(inputs=[state_input, turn_input], outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch_indices = random.sample(range(len(self.replay_buffer)), self.batch_size)
        minibatch = [self.replay_buffer[i] for i in minibatch_indices]
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        valid_actions = [i[5] for i in minibatch]
        turns = np.array([i[6] for i in minibatch])
        
        target = self.model.predict([states.reshape(-1,8,8,1),turns],verbose=None)
        Qvalue = self.target_model.predict([next_states.reshape(-1,8,8,1),turns],verbose=None)
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i] /100.0
            else:
                target[i][actions[i]] = rewards[i] /100.0 + self.gamma * np.amax(Qvalue[i][valid_actions[i]])
            self._InsertHashTable(states[i])
        loss = self.model.train_on_batch([states.reshape(-1,8,8,1),turns], target)
        for index in sorted(minibatch_indices, reverse=True):
            del self.replay_buffer[index]
        return loss

    def BehaviorPolicy(self, env,state,turn,valid_action):
        q_values = self.target_model.predict([state.reshape(-1,8,8,1),np.array([float(turn)])],verbose=None)[0][valid_action]
        count = []
        for i in range(len(valid_action)):
            count.append(self.GetCount(env.simulateNextState(valid_action[i])))
        if(sum(count)==0):
            return random.choice(valid_action)
        else:
            uct_values = q_values+ np.sqrt(2*np.log(np.sum(count))/(1+np.array(count)))
            return valid_action[np.argmax(uct_values)]
    
    def EstimatePolicy(self, state ,turn, valid_action):
        q_values = self.target_model.predict([state.reshape(-1,8,8,1),np.array([float(turn)])],verbose=None)[0]
        mask = np.zeros_like(q_values, dtype=bool)
        mask[valid_action] = True
        return valid_action[np.argmax(q_values[mask])]
    
    def InsertBuffer(self, state, action, reward,next_states,done,valid_actions,turns):
        if len(self.replay_buffer) > self.maxBufferSize:
            self.replay_buffer.pop(0)
        self.replay_buffer.append([state, action, reward,next_states,done,valid_actions,turns])

    def GetCount(self, state):
        hash_value = self._Gethash(state)
        half_length = len(hash_value) // 2
        first_half = hash_value[:half_length]
        second_half = hash_value[half_length:]
        row = int(first_half, 16) % 10
        col = int(second_half, 16) % 10
        if hash_value not in self.hashtable[row][col]:
            return 0
        else:
            return self.hashtable[row][col][hash_value]

    def flush(self):
        buffer = self.replay_buffer
        self.replay_buffer = []
        return buffer

    def _Gethash(self, state):
        # 2차원 배열의 각 요소 값을 결합하여 하나의 문자열로 만듭니다.
        combined_string = ''.join(map(str, state))
        # 결합된 문자열에 대한 해시 값을 계산합니다.
        hash_value = hashlib.md5(combined_string.encode()).hexdigest()
        return hash_value

    def _InsertHashTable(self, state):
        hash_value = self._Gethash(state)
        half_length = len(hash_value) // 2
        first_half = hash_value[:half_length]
        second_half = hash_value[half_length:]
        row = int(first_half, 16) % 10
        col = int(second_half, 16) % 10
        if hash_value not in self.hashtable[row][col]:
            self.hashtable[row][col][hash_value] = 1
        else:
            self.hashtable[row][col][hash_value] += 1