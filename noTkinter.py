import math
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import datetime

# Ball 클래스 생성 : player, ball(들) 생성 시 사용
class Ball:
    def __init__(self, x, y, r, speed_x, speed_y):
        self.x = x                              # 중심 좌표
        self.y = y
        self.r = r                              # 반지름
        self.speed_x = speed_x                  # 속도
        self.speed_y = speed_y

    def Move(self, width, height):
        # 현재 속도(speed_x, speed_y)에 따라 이동한다.
        self.x += self.speed_x
        self.y += self.speed_y
        # 벽과의 충돌 체크
        self.CheckCollisionWall(width, height)

    def CheckCollisionWall(self, width, height):
        # 왼쪽 또는 오른쪽 경계를 넘으면 x 속도의 부호를 반전시킨다.
        if self.x - self.r <= 0 or self.x + self.r >= width:
            self.speed_x = -self.speed_x
        # 위 또는 아래 경계를 넘으면 y 속도의 부호를 반전시킨다.
        if self.y - self.r <= 0 or self.y + self.r >= height:
            self.speed_y = -self.speed_y

    def CheckCollisionBall(self, ball):
        distance = math.sqrt((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2)
        if distance < self.r + ball.r:
            return True
        return False

    def SetSpeed(self, speed_x, speed_y):
        self.speed_x = speed_x
        self.speed_y = speed_y

class MovingBallsEnv:
    def __init__(self):
        self.ball_count = 5
        self.state_size = 4 + self.ball_count * 4       # 인공 신경망 입력 : player의 중심좌표와 방향(x,y,speed_x,speed_y) + 공 개수 * 각 공의 중심좌표와 방향 (x,y,speed_x,speed_y)
        self.action_size = 4                            # 출력 : 방향 전환 (0, 1, 2, 3 => 동, 서, 남, 북)+
        self.width = 600
        self.height = 400
        self.player = None

    def Reset(self):
        # player 생성
        self.player = Ball(400, 200, 25, 10, 0)

        # ball들 생성
        self.balls = []
        for i in range(self.ball_count):
            self.balls.append(Ball(100, 100, 25, random.randint(1, 10), random.randint(1, 10)))

        return self.MakeState()

    # 주기적으로 이동
    def Move(self):
        done = False
        reward = 1
        self.player.Move(self.width, self.height)  # player 이동
        if self.player.CheckCollisionWall(self.width, self.height):
            done = True
            reward = -100  # 벽과 충돌하면 큰 페널티
        for ball in self.balls:              # ball들 이동
            ball.Move(self.width, self.height)
        for ball in self.balls:
            if self.player.CheckCollisionBall(ball):
                done = True
                reward = -100  # 공과 충돌해도 큰 페널티
                break
        return self.MakeState(), reward, done

    def MakeState(self):
        state = []
        state.extend([self.player.x, self.player.y, self.player.speed_x, self.player.speed_y])
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.speed_x, ball.speed_y])
        return np.array(state)

    def Step(self, action):
        if action == 0:
            self.player.SetSpeed(10, 0)
        elif action == 1:
            self.player.SetSpeed(-10, 0)
        elif action == 2:
            self.player.SetSpeed(0, 10)
        elif action == 3:
            self.player.SetSpeed(0, -10)

        return self.Move()

# DQN 에이전트 구현
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 할인율
        self.epsilon = 1.0   # 탐험율
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.scores = []  # 점수 저장

    def _build_model(self):
        # 인공신경망 모델
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_score(self, score):
        self.scores.append(score)
        current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        log_message = f"[{current_time}] New Score: {score}, Best Score: {max(self.scores)}, Average Score: {np.mean(self.scores)}\n"

        # 파일에 로그 메시지 저장
        with open("training_log.txt", "a") as log_file:
            log_file.write(log_message)

# 환경 만들기
env = MovingBallsEnv()
agent = DQNAgent(env.state_size, env.action_size)
batch_size = 32

def main():
    state = env.Reset()
    state = np.reshape(state, [1, env.state_size])
    score = 0

    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.Step(action)
        next_state = np.reshape(next_state, [1, env.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            agent.save_score(score)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # 에피소드 끝난 후에만 학습
            score = 0
            state = env.Reset()
            state = np.reshape(state, [1, env.state_size])

if __name__ == "__main__":
    main()