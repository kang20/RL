import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses

learning_rate = 0.0002
gamma = 0.98
n_rollout = 10 # 10개를 모아서 업데이트


class ActorCritic(tf.keras.Model):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = layers.Dense(256, activation='relu')  # 첫 번째 완전연결층
        self.fc_pi = layers.Dense(2, activation='softmax')  # 정책 네트워크의 출력층
        self.fc_v = layers.Dense(1)  # 가치 네트워크의 출력층
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)  # Adam 옵티마이저

    def pi(self, x):
        x = self.fc1(x)
        prob = self.fc_pi(x)  # 행동 확률
        return prob

    def v(self, x):
        x = self.fc1(x)
        v = self.fc_v(x)  # 상태 가치
        return v

    def put_data(self, transition):
        self.data.append(transition)  # 샘플 데이터 저장

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch = tf.convert_to_tensor(np.array(s_lst), dtype=tf.float32)
        a_batch = tf.convert_to_tensor(np.array(a_lst), dtype=tf.int32)
        r_batch = tf.convert_to_tensor(np.array(r_lst), dtype=tf.float32)
        s_prime_batch = tf.convert_to_tensor(np.array(s_prime_lst), dtype=tf.float32)
        done_batch = tf.convert_to_tensor(np.array(done_lst), dtype=tf.float32)

        self.data = []  # 데이터 비우기
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        with tf.GradientTape() as tape:
            td_target = r + gamma * self.v(s_prime) * done
            delta = td_target - self.v(s)
            pi = self.pi(s)
            pi_a = tf.gather_nd(pi, a, batch_dims=1)
            loss = -tf.reduce_mean(tf.math.log(pi_a) * delta) + tf.reduce_mean(losses.Huber()(td_target, self.v(s)))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # 네트워크 가중치 업데이트


def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()
        while not done:
            for t in range(n_rollout):
                s_tensor = tf.convert_to_tensor(np.array([s]), dtype=tf.float32)
                prob = model.pi(s_tensor)
                a = np.random.choice(range(prob.shape[1]), p=prob.numpy()[0])  # 행동 선택
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()