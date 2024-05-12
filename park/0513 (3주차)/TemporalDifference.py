import random


class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def act(self, rnd):
        if rnd == 0:
            self.moveLeft()
        elif rnd == 1:
            self.moveRight()
        elif rnd == 2:
            self.moveUp()
        else:
            self.moveDown()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def moveLeft(self):
        if self.x > 0:
            self.x -= 1

    def moveRight(self):
        if self.x < 3:
            self.x += 1

    def moveUp(self):
        if self.y > 0:
            self.y -= 1

    def moveDown(self):
        if self.y < 3:
            self.y += 1

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False

    def getState(self):
        return (self.x, self.y)

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)


class Agent():
    def __init__(self):
        pass

    def select_action(self):
        probability = random.random()
        action = 0
        if probability < 0.25:
            action = 0
        elif probability < 0.5:
            action = 1
        elif probability < 0.75:
            action = 2
        else:
            action = 3
        return action


def main():
    env = GridWorld()
    agent = Agent()
    data = [[0.0] * 4 for _ in range(4)]
    gamma = 1.0
    alpha = 0.01

    for i in range(50000):
        done = False
        while not done:
            x, y = env.getState()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.act(action)
            x_prime, y_prime = env.getState()

            data[x][y] = (data[x][y] + alpha
                          * (reward + gamma * data[x_prime][y_prime] - data[x][y]))

        env.reset()

    for row in data:
        print([f"{value:.1f}" for value in row])


if __name__ == "__main__":
    main()
