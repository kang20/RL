import random
import numpy as np # q(s, a) 관리
from GridWorld import GridWorld

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4)) # q 밸류를 저장하는 변수, 0으로 초기화함
        self.eps = 0.9 # 앱실론, 점차 줄어들 예정
        self.alpha = 0.01

    def select_action(self, s):
        # 앱실론 그리디로 액션을 선택함
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아,
        # q 테이블의 값을 업데이트
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y = s
            # 몬테카를로 방식을 이용해 테이블 업데이트
            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (cum_reward - self.q_table[x, y, a])
            cum_reward = cum_reward + r

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_list = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)

def main():
    env = GridWorld()
    agent = QAgent()
    count = 1000

    for n_epi in range(count):
        done = False
        history = []

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        print(n_epi, "번째 에피소드 반복")
        agent.show_table()
        print()
        agent.update_table(history) # 한 에피소드 끝나면 history로 에이전트 업데이트
        agent.anneal_eps() # 앱실론 0.03씩 감소

    print(count, "번째 에피소드 반복")
    agent.show_table()

if __name__ == "__main__":
    main()