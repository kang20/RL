from agent import Agent
from gridworld import GridWorld
import math
import time

def main():
    # 실험 변수
    data_n = 3 # grid size
    print_n = 5000 # 출력 주기

    env = GridWorld(data_n)
    agent = Agent() 
    data = [[0.0] * data_n for _ in range(data_n)]
    gamma = 1.0
    alpha = 0.0001

    for k in range(500000):
        done = False
        history = []
        while not done:
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x, y, reward))
        env.reset()

        cum_reward = 0 #Gt
        for transition in history[::-1]: #뒤부터 계산
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            cum_reward = reward + gamma * cum_reward #Gt = R(t+1) + gammaG(T+1)

        if k % print_n == 0:
            print("epoch:  ",k,"-----------------")
            for row in data:
                print(row)

    print("epoch:  ",k,"-----------------")
    for row in data:
        print(row)
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f"{end - start:.5f} sec")