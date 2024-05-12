import random

class GridWorld():
    def __init__(self,grid_size):
        self.x = 0
        self.y = 0
        self.grid_size = grid_size

    def step(self, a):
        if a == 0:
            self.move_right()
        elif a == 1:
            self.move_left()
        elif a == 2:
            self.move_up()
        elif a == 3:
            self.move_down()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1
        if self.y > self.grid_size-1:
            self.y = self.grid_size-1

    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

    def move_down(self):
        self.x += 1
        if self.x > self.grid_size-1:
            self.x = self.grid_size-1

    def is_done(self):
        if self.x == self.grid_size-1 and self.y == self.grid_size-1:
            return True
        else:
            return False

    def get_state(self):
        return (self.x, self.y)

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)