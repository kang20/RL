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
        self.fc1 = layers.Dense(256, activation='relu')  # 첫 번째 완전 연결층
        self.fc_pi = layers.Dense(2, activation='softmax')  # 정책 네트워크의 출력층
        self.fc_v = layers.Dense(1)  # 가치 네트워크의 출력층
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)  # Adam 옵티마이저

    # 상태 s를 받아서 정책 네트워크를 통해 액션별 선택 확률 값을 리턴
    # 정책 확률 계산
    def pi(self, x):
        x = self.fc1(x)
        prob = self.fc_pi(x)  # 행동 확률
        return prob

    # 상태 s를 받아서 가치 네트워크를 통해 상태 가치를 리턴
    def v(self, x):
        x = self.fc1(x)
        v = self.fc_v(x)  # 상태 가치
        return v

    def put_data(self, transition):
        self.data.append(transition)  # 샘플 데이터 저장

    def make_batch(self):
        # 배치 데이터를 담을 리스트 초기화
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0]) # 보상 스케일 조정
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 # 종료 여부를 마스크로 변환
            done_lst.append([done_mask])

        # 리스트를 텐서로 변환
        s_batch = tf.convert_to_tensor(np.array(s_lst), dtype=tf.float32)
        a_batch = tf.convert_to_tensor(np.array(a_lst), dtype=tf.int32)
        r_batch = tf.convert_to_tensor(np.array(r_lst), dtype=tf.float32)
        s_prime_batch = tf.convert_to_tensor(np.array(s_prime_lst), dtype=tf.float32)
        done_batch = tf.convert_to_tensor(np.array(done_lst), dtype=tf.float32)

        self.data = []  # 데이터 비우기
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()  # 배치 생성
        with tf.GradientTape() as tape:
            td_target = r + gamma * self.v(s_prime) * done  # TD 타겟 계산
            delta = td_target - self.v(s)  # TD 오류 계산
            pi = self.pi(s)  # 정책 확률 계산
            pi_a = tf.gather_nd(pi, a, batch_dims=1)  # 선택된 행동의 확률 추출

            # 정책 손실과 가치 손실의 결합된 손실 함수 (MSE 사용)
            policy_loss = -tf.reduce_mean(tf.math.log(pi_a) * delta)  # 정책 손실
            value_loss = tf.reduce_mean(tf.square(td_target - self.v(s)))  # 가치 손실 (MSE)
            loss = policy_loss + value_loss  # 전체 손실

        grads = tape.gradient(loss, self.trainable_variables)  # 손실에 대한 그래디언트 계산
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # 모델 파라미터 업데이트

def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic() # ActorCritic 모델 생성
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset() # 환경 초기화
        while not done:
            for t in range(n_rollout): # 10개를 모아서 업데이트
                s_tensor = tf.convert_to_tensor(np.array([s]), dtype=tf.float32) # 상태를 텐서로 변환
                prob = model.pi(s_tensor)  # 정책 네트워크로부터 행동 확률 계산
                a = np.random.choice(range(prob.shape[1]), p=prob.numpy()[0]) # 확률에 따라 행동 선택
                s_prime, r, done, info = env.step(a) # 환경에서 다음 상태, 보상 받기
                model.put_data((s, a, r, s_prime, done))  # 데이터 저장
                s = s_prime
                score += r

                if done:
                    break
            model.train_net() # 네트워크 학습
            
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()