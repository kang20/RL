import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.0002
gamma = 0.98
n_rollout = 10 # 10개를 모아서 업데이트


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
                prob = model.pi(torch.from_numpy(s).float())  # 정책 네트워크에서 행동 확률 계산
                m = Categorical(prob)
                a = m.sample().item()  # 행동 샘플링
                s_prime, r, done, info = env.step(a)  # 환경에서 다음 상태, 보상 받기
                model.put_data((s, a, r, s_prime, done))  # 데이터 저장
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()  # 네트워크 학습
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)  # 첫 번째 완전연결층
        self.fc_pi = nn.Linear(256, 2)  # 정책 네트워크의 출력층
        self.fc_v = nn.Linear(256, 1)  # 가치 네트워크의 출력층
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Adam 옵티마이저

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)  # 행동 확률 계산
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)  # 상태 가치 계산
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

        s_batch = torch.tensor(s_lst, dtype=torch.float32)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst, dtype=torch.float32)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float32)
        done_batch = torch.tensor(done_lst, dtype=torch.float32)

        self.data = []  # 데이터 비우기
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())  # 손실 계산
        self.optimizer.zero_grad()
        loss.mean().backward()  # 역전파
        self.optimizer.step()  # 네트워크 가중치 업데이트


if __name__ == "__main__":
    main()