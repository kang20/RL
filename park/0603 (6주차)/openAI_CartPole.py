import gym  # OpenAI GYM 라이브러리
import collections  # deque 활용
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 하이퍼 파라미터 정의
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

# 리플레이 버퍼 클래스
class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (np.array(s_lst, dtype=np.float32), np.array(a_lst, dtype=np.int32),
                np.array(r_lst, dtype=np.float32), np.array(s_prime_lst, dtype=np.float32),
                np.array(done_mask_lst, dtype=np.float32))

    def size(self):
        return len(self.buffer)

# Q밸류 네트워크 클래스
class Qnet(models.Model):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(2, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.call(obs)
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return np.argmax(out.numpy(), axis=1)[0]

# 학습 함수
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        with tf.GradientTape() as tape:
            q_out = q(s)
            q_a = tf.gather_nd(q_out, indices=np.expand_dims(a, axis=1), batch_dims=1)
            max_q_prime = tf.reduce_max(q_target(s_prime), axis=1, keepdims=True)
            target = r + gamma * max_q_prime * done_mask
            loss = tf.reduce_mean(tf.square(target - q_a))

        grads = tape.gradient(loss, q.trainable_variables)
        optimizer.apply_gradients(zip(grads, q.trainable_variables))

# 메인 함수
def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()

    # 모델 빌드
    q.build(input_shape=(None, 4))
    q_target.build(input_shape=(None, 4))

    q_target.set_weights(q.get_weights())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optimizers.Adam(learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(np.array([s], dtype=np.float32), epsilon)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.set_weights(q.get_weights())
            print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()