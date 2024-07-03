import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# 하이퍼파라미터 설정
learning_rate = 0.0002
gamma = 0.98


def main():
    env = gym.make('CartPole-v1')  # CartPole 환경 생성
    pi = Policy()  # Policy 신경망 생성
    score = 0.0  # 점수 초기화
    print_interval = 20  # 출력 간격 설정

    for n_epi in range(10000):  # 10000번의 에피소드 동안 반복
        s = env.reset()  # 환경 초기화 및 초기 상태 얻기
        done = False  # 에피소드 완료 여부 초기화

        while not done:  # 에피소드가 끝날 때까지 반복
            prob = pi(torch.from_numpy(s).float())  # 상태 s에 대한 행동 확률 계산
            m = Categorical(prob)  # 카테고리 분포 생성
            a = m.sample()  # 행동 샘플링
            s_prime, r, done, info = env.step(a.item())  # 행동 a를 환경에 적용하고 다음 상태, 보상, 완료 여부 얻기
            pi.put_data((r, prob[a]))  # 보상과 확률 저장
            s = s_prime  # 다음 상태로 전이
            score += r  # 점수 갱신

            if done:  # 에피소드가 끝나면
                break

        pi.train_net()  # 신경망 학습

        if n_epi % print_interval == 0 and n_epi != 0:  # 일정 에피소드마다 평균 점수 출력
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0  # 점수 초기화

    env.close()  # 환경 종료


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []  # 데이터를 저장할 리스트
        self.fc1 = nn.Linear(4, 128)  # 첫 번째 완전 연결 레이어
        self.fc2 = nn.Linear(128, 2)  # 두 번째 완전 연결 레이어
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Adam 최적화 알고리즘 설정

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 첫 번째 레이어를 통과한 후 ReLU 활성화 함수 적용
        x = F.softmax(self.fc2(x), dim=0)  # 두 번째 레이어를 통과한 후 소프트맥스 함수 적용
        return x

    def put_data(self, item):
        self.data.append(item)  # 데이터를 리스트에 저장

    def train_net(self):
        R = 0  # 누적 보상 초기화
        self.optimizer.zero_grad()  # 그래디언트 초기화

        for r, prob in self.data[::-1]:  # 저장된 데이터를 역순으로 반복
            R = r + gamma * R  # 누적 보상 계산
            loss = -R * torch.log(prob)  # 손실 계산
            loss.backward()  # 역전파를 통해 그래디언트 계산

        self.optimizer.step()  # 최적화 적용
        self.data = []  # 데이터 초기화


if __name__ == "__main__":
    main()
