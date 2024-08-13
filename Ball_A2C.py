import tkinter as tk
import math
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
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

# MovingBallsEnv 클래스 정의
class MovingBallsEnv:
    def __init__(self, window):
        self.ball_count = 5
        self.state_size = 4 + self.ball_count * 4        # 인공 신경망 입력 : player의 중심좌표와 방향(x,y,speed_x,speed_y) + 공 개수 * 각 공의 중심좌표와 방향 (x,y,speed_x,speed_y)
        self.action_size = 4                             # 행동 공간: 동, 서, 남, 북

        self.window = window
        self.canvas = tk.Canvas(self.window, width=600, height=400, background="white")
        self.canvas.pack(expand=1, fill=tk.BOTH)
        self.player = None

    # 환경을 초기화하고, 새로운 상태를 반환
    def Reset(self):
        self.Clear()

        self.player = Ball(self.canvas, "green", 400, 200, 25, 0, 0)
        self.balls = []
        for i in range(self.ball_count):
            self.balls.append(Ball(self.canvas, "red", 100, 100, 25,
                                   random.randint(1, 10), random.randint(1, 10)))
        return self.MakeState()

    # 플레이어와 공들을 이동시키고, 보상과 종료 상태를 계산
    def Move(self):
        done = False
        reward = 1

        # 벽과의 거리 계산
        min_distance_to_wall = min(self.player.x, self.canvas.winfo_width() - self.player.x,
                                   self.player.y, self.canvas.winfo_height() - self.player.y)

        # 벽과 너무 가까워지면 페널티 부여
        if min_distance_to_wall < self.player.r * 2:
            reward -= 100

        # 벽에서 멀어질 때 보상 추가
        if min_distance_to_wall > self.player.r * 5:
            reward += 50  # 벽에서 멀어질수록 보상 증가

        # 공들과의 거리 합 계산
        distance_sum = sum(
            [math.sqrt((self.player.x - ball.x) ** 2 + (self.player.y - ball.y) ** 2) for ball in self.balls])

        self.player.Move()
        if self.player.CheckCollisionWall():
            done = True
            reward = -300  # 벽과 충돌하면 큰 페널티

        for ball in self.balls:
            ball.Move()
            ball.CheckCollisionWall()

        # 새로 계산한 공들과의 거리 합 계산
        new_distance_sum = sum(
            [math.sqrt((self.player.x - ball.x) ** 2 + (self.player.y - ball.y) ** 2) for ball in self.balls])

        # 거리가 멀어지면 보상, 가까워지면 페널티
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

    # 현재 환경을 초기화 (공과 플레이어 제거)
    def Clear(self):
        if self.player:
            self.player.Delete()
            for ball in self.balls:
                ball.Delete()

    # 현재 상태를 벡터로 변환하여 반환
    def MakeState(self):
        state = [self.player.x, self.player.y, self.player.speed_x, self.player.speed_y]
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.speed_x, ball.speed_y])
        return np.array(state)

    # 주어진 행동에 따라 플레이어의 속도를 설정하고, 환경을 업데이트
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
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0  # 초기 탐험 확률
        self.epsilon_min = 0.01  # 최소 탐험 확률
        self.epsilon_decay = 0.995  # 탐험 확률 감소 비율
        self.learning_rate = 0.001
        self.memory = deque(maxlen=1000)  # 경험 리플레이 메모리
        self.batch_size = 16  # 배치 크기
        self.actor, self.critic = self._build_model()
        self.scores = []
        self.step = []  # 스텝 저장

    def _build_model(self):
        state_input = Input(shape=(self.state_size,))

        # Actor 네트워크
        actor_hidden = Dense(128, activation='relu')(state_input)
        actor_hidden = Dropout(0.3)(actor_hidden)
        actor_hidden = Dense(64, activation='relu')(actor_hidden)
        actor_hidden = Dropout(0.3)(actor_hidden)
        actor_output = Dense(self.action_size, activation='softmax')(actor_hidden)
        actor_model = Model(inputs=state_input, outputs=actor_output)
        actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate))

        # Critic 네트워크
        critic_hidden = Dense(128, activation='relu')(state_input)
        critic_hidden = Dropout(0.3)(critic_hidden)
        critic_hidden = Dense(64, activation='relu')(critic_hidden)
        critic_hidden = Dropout(0.3)(critic_hidden)
        critic_output = Dense(1, activation='linear')(critic_hidden)
        critic_model = Model(inputs=state_input, outputs=critic_output)
        critic_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return actor_model, critic_model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # 무작위로 행동을 선택할 확률
            return np.random.choice(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        policy = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=policy)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])

            value = self.critic.predict(state, verbose=0)
            next_value = self.critic.predict(next_state, verbose=0)

            if done:
                advantage = reward - value
                target = np.array([[reward]])  # Critic을 위한 target
            else:
                advantage = reward + self.gamma * next_value - value
                target = reward + self.gamma * next_value

            actions_onehot = np.zeros([1, self.action_size])
            actions_onehot[0, action] = 1

            self.actor.fit(state, actions_onehot, sample_weight=advantage.flatten(), verbose=0)
            self.critic.fit(state, target, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.replay()

    def save_score(self, score, step):
        self.scores.append(score)
        self.step.append(step)
        current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        log_message = f"[{current_time}] New Score: {score}, Best Score: {max(self.scores)}, Average Score: {np.mean(self.scores)}\n step num : {step} , max step num : {max(self.step)}, average step num : {np.mean(self.step)}\n"

        with open("A2C_Training_Log.txt", "a") as log_file:
            log_file.write(log_message)
def main():
    global env, agent
    done = False
    state = env.Reset()
    state = np.reshape(state, [1, env.state_size])
    score = 0
    step_n = 0

    def step():
        nonlocal state, score, done , step_n

        action = agent.act(state)
        next_state, reward, done = env.Step(action)
        agent.train(state, action, reward, next_state, done)
        state = np.reshape(next_state, [1, env.state_size])
        score += reward
        step_n += 1

        if done:
            agent.save_score(score, step_n)
            score = 0
            step_n = 0
            state = env.Reset()
            state = np.reshape(state, [1, env.state_size])

        window.after(50, step)

    step()


# 환경 만들기
window = tk.Tk()
env = MovingBallsEnv(window)
agent = A2CAgent(env.state_size, env.action_size)

window.after(1000, main)
window.mainloop()

window.after(1000, main)
window.mainloop()