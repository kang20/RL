import tkinter as tk
import math
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense
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

    def Reset(self):
        self.Clear()

        self.player = Ball(self.canvas, "green", 400, 200, 25, 0, 0)
        self.balls = []
        for i in range(self.ball_count):
            self.balls.append(Ball(self.canvas, "red", 100, 100, 25,
                                   random.randint(1, 10), random.randint(1, 10)))
        return self.MakeState()

    # 보상 구조 수정
    def Move(self):
        done = False
        reward = 1

        previous_distance_sum = sum(
            [math.sqrt((self.player.x - ball.x) ** 2 + (self.player.y - ball.y) ** 2) for ball in self.balls])

        self.player.Move()
        if self.player.CheckCollisionWall():
            done = True
            reward = -100  # 벽과 충돌하면 큰 페널티

        for ball in self.balls:
            ball.Move()
            ball.CheckCollisionWall()

        if not done:
            # 현재 상태에서 빨간 공들과의 거리 합을 계산
            current_distance_sum = sum(
                [math.sqrt((self.player.x - ball.x) ** 2 + (self.player.y - ball.y) ** 2) for ball in self.balls])

            # 만약 거리가 증가했다면, 보상을 증가
            if current_distance_sum > previous_distance_sum:
                reward += 10

        for ball in self.balls:
            if self.player.CheckCollisionBall(ball):
                done = True
                reward = -100  # 공과 충돌해도 큰 페널티
                break

        return self.MakeState(), reward, done


    def Clear(self):
        if self.player:
            self.player.Delete()
            for ball in self.balls:
                ball.Delete()

    def MakeState(self):
        state = [self.player.x, self.player.y, self.player.speed_x, self.player.speed_y]
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.speed_x, ball.speed_y])
        return np.array(state)

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

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 0.1  # 탐험 확률 설정
        self.learning_rate = 0.001
        self.actor, self.critic = self._build_model()
        self.scores = []

    def _build_model(self):
        # 공통 입력
        state_input = Input(shape=(self.state_size,))

        # Actor 네트워크
        actor_hidden = Dense(24, activation='relu')(state_input)
        actor_output = Dense(self.action_size, activation='softmax')(actor_hidden)
        actor_model = Model(inputs=state_input, outputs=actor_output)
        actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate))

        # Critic 네트워크
        critic_hidden = Dense(24, activation='relu')(state_input)
        critic_output = Dense(1, activation='linear')(critic_hidden)
        critic_model = Model(inputs=state_input, outputs=critic_output)
        critic_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return actor_model, critic_model

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # 무작위로 행동을 선택할 확률
            return np.random.choice(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        policy = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=policy)

    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        value = self.critic.predict(state, verbose=0)
        next_value = self.critic.predict(next_state, verbose=0)

        if done:
            advantage = reward - value
            target = np.array([[reward]])  # Critic을 위한 target
        else:
            advantage = reward + self.gamma * next_value - value # Critic의 예측
            target = reward + self.gamma * next_value

        actions_onehot = np.zeros([1, self.action_size])
        actions_onehot[0, action] = 1

        self.actor.fit(state, actions_onehot, sample_weight=advantage.flatten(), verbose=0)
        self.critic.fit(state, target, verbose=0)

    def save_score(self, score):
        self.scores.append(score)
        current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        log_message = f"[{current_time}] New Score: {score}, Best Score: {max(self.scores)}, Average Score: {np.mean(self.scores)}\n"

        # 파일에 로그 메시지 저장
        with open("A2C_Training_Log.txt", "a") as log_file:
            log_file.write(log_message)

def main():
    global env, agent
    done = False
    state = env.Reset()
    state = np.reshape(state, [1, env.state_size])
    score = 0

    def step():
        nonlocal state, score, done

        action = agent.act(state)
        next_state, reward, done = env.Step(action)
        agent.train(state, action, reward, next_state, done)
        state = np.reshape(next_state, [1, env.state_size])
        score += reward

        if done:
            agent.save_score(score)
            score = 0
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