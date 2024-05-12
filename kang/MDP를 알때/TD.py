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
    alpha = 0.01

    for k in range(500000):
        done = False
        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.step(action)
            x_prime, y_prime = env.get_state()

            data[x][y] = data[x][y] + alpha*(reward+gamma*data[x_prime][y_prime]-data[x][y])

            if k % print_n == 0:
                print("epoch:  ",k,"-----------------")
                for row in data:
                    print(row)

        env.reset()
    print("epoch:  ",k,"-----------------") 
    for row in data:
        print(row)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f"{end - start:.5f} sec")