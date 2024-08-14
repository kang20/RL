import tkinter as tk
import math
import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime

# Ball 클래스 생성 : player, ball(들) 생성 시 사용
class Ball:
    def __init__(self, canvas, color, x, y, r, speed_x, speed_y):
        self.canvas = canvas
        self.color = color                      # 색깔
        self.x = x                              # 중심 좌표
        self.y = y
        self.r = r                              # 반지름
        self.speed_x = speed_x                  # 속도
        self.speed_y = speed_y
        self.id = canvas.create_oval(x - r, y - r, x + r, y + r, fill=color)


    def Move(self):
        self.canvas.move(self.id, self.speed_x, self.speed_y)
        self.canvas.update()
        (x1, y1, x2, y2) = self.canvas.coords(self.id)
        self.x, self.y = x1 + self.r, y1 + self.r

    def CheckCollisionWall(self):
        collision = False
        if self.x - self.r <= 0 or self.x + self.r >= self.canvas.winfo_width():
            self.speed_x = -self.speed_x
            collision = True
        if self.y - self.r <= 0 or self.y + self.r >= self.canvas.winfo_height():
            self.speed_y = -self.speed_y
            collision = True
        return collision

    def CheckCollisionBall(self, ball):
        distance = math.sqrt((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2)
        if distance < self.r + ball.r:
            return True
        return False

    def SetSpeed(self, speed_x, speed_y):
        self.speed_x = speed_x
        self.speed_y = speed_y

    def Delete(self):
        self.canvas.delete(self.id)

class MovingBallsEnv:
    def __init__(self, window):
        self.ball_count = 5
        self.state_size = 4 + self.ball_count * 4  # 상태 크기
        self.action_size = 4  # 행동 공간 크기: 동, 서, 남, 북
        self.window = window
        self.canvas = tk.Canvas(self.window, width=600, height=400, background="white")
        self.canvas.pack(expand=1, fill=tk.BOTH)
        self.player = None

    # 환경 초기화 및 초기 상태 반환
    def Reset(self):
        self.Clear()
        self.player = Ball(self.canvas, "green", 400, 200, 25, 10, 0)
        self.balls = []
        for i in range(self.ball_count):
            self.balls.append(Ball(self.canvas, "red", 100, 100, 25, random.randint(1, 10), random.randint(1, 10)))
        return self.MakeState()

    # 현재 캔버스에서 모든 공 제거
    def Clear(self):
        if self.player:
            self.player.Delete()
            for ball in self.balls:
                ball.Delete()

    # 플레이어와 공들을 움직이고, 충돌 여부 확인 및 보상 계산
    def Move(self):
        done = False
        reward = 1

        # 벽과의 거리 계산
        min_distance_to_wall = min(self.player.x, self.canvas.winfo_width() - self.player.x,
                                   self.player.y, self.canvas.winfo_height() - self.player.y)

        # 벽과 너무 가까워지면 페널티 부여
        if min_distance_to_wall < self.player.r * 2:
            reward -= 50

        distance_sum = sum(
            [math.sqrt((self.player.x - ball.x) ** 2 + (self.player.y - ball.y) ** 2) for ball in self.balls])

        self.player.Move()
        if self.player.CheckCollisionWall():
            done = True
            reward = -100  # 벽과 충돌하면 큰 페널티

        for ball in self.balls:
            ball.Move()
            ball.CheckCollisionWall()

        # 빨간 공과의 거리 계산
        new_distance_sum = sum(
            [math.sqrt((self.player.x - ball.x) ** 2 + (self.player.y - ball.y) ** 2) for ball in self.balls])

        if new_distance_sum > distance_sum:
            reward += 50  # 공들과의 거리가 멀어지면 보상 증가
        else:
            reward -= 10  # 공들과의 거리가 가까워지면 페널티

        for ball in self.balls:
            if self.player.CheckCollisionBall(ball):
                done = True
                reward = -100  # 공과 충돌해도 큰 페널티
                break

        return self.MakeState(), reward, done

    # 현재 상태 정보를 생성하여 반환
    def MakeState(self):
        state = [self.player.x, self.player.y, self.player.speed_x, self.player.speed_y]
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.speed_x, ball.speed_y])
        return np.array(state)

    # 주어진 행동(action)에 따라 플레이어를 이동시키고 환경을 업데이트
    def Step(self, action):
        if action == 0:  # 동
            self.player.SetSpeed(10, 0)
        elif action == 1:  # 서
            self.player.SetSpeed(-10, 0)
        elif action == 2:  # 남
            self.player.SetSpeed(0, 10)
        elif action == 3:  # 북
            self.player.SetSpeed(0, -10)

        return self.Move()

# DQN 에이전트 구현
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100)
        self.gamma = 0.95    # 감마 값
        self.epsilon = 1.0   # 탐험율
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.scores = []  # 점수 저장
        self.step = [] # 스텝 저장

    # 인공신경망 모델
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # 경험을 메모리에 저장
    # replay 메소드에서 과거 샘플링할 때 씀
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 현재 상태에 따라 행동 결정
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    # 저장된 경험으로 학습
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

    def save_score(self, score, step):
        self.scores.append(score)
        self.step.append(step)
        current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        log_message = f"[{current_time}] New Score: {score}, Best Score: {max(self.scores)}, Average Score: {np.mean(self.scores)}\n step num : {step} , max step num : {max(self.step)}, average step num : {np.mean(self.step)}\n"

        with open("DeepQLearning_Training_Log.txt", "a") as log_file:
            log_file.write(log_message)

def main():
    global env, agent      # 환경 및 에이전트
    done = False
    state = env.Reset()
    state = np.reshape(state, [1, env.state_size])
    score = 0
    step_n = 0

    def step():
        nonlocal state, score, done , step_n

        action = agent.act(state)
        next_state, reward, done = env.Step(action)
        next_state = np.reshape(next_state, [1, env.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        step_n += 1

        if done:
            agent.save_score(score, step_n)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # 에피소드 끝난 후에만 학습
            score = 0
            step_n = 0
            state = env.Reset()
            state = np.reshape(state, [1, env.state_size])

        window.after(50, step)  # 50ms 후 step 함수 호출

    step()

window = tk.Tk()
env = MovingBallsEnv(window)

agent = DQNAgent(env.state_size, env.action_size)
batch_size = 32

window.after(1000, main)
window.mainloop()