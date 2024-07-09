import gym
import tensorflow as tf
import numpy as np

# 하이퍼파라미터 설정
learning_rate = 0.0002
gamma = 0.98


class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(2, activation='softmax')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        with tf.GradientTape() as tape:
            loss = 0
            for r, prob in self.data[::-1]:
                R = r + gamma * R
                loss += -R * tf.math.log(prob)

        # Ensure that the gradients are calculated for the trainable variables
        grads = tape.gradient(loss, self.trainable_variables)

        # Only apply gradients if they are not None
        grads_and_vars = [(grad, var) for grad, var in zip(grads, self.trainable_variables) if grad is not None]

        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)

        self.data = []


def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        s = np.reshape(s, [1, 4])
        done = False

        while not done:
            prob = pi(tf.convert_to_tensor(s, dtype=tf.float32))
            prob = prob.numpy()
            a = np.random.choice(len(prob[0]), p=prob[0])
            s_prime, r, done, info = env.step(a)
            s_prime = np.reshape(s_prime, [1, 4])
            pi.put_data((r, prob[0][a]))
            s = s_prime
            score += r

            if done:
                break

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == "__main__":
    main()