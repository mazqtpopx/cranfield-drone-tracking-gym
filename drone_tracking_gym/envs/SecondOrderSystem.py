import numpy as np

class SecondOrderSystem():
    def __init__(self, f, z, r, x0, dt):
        self.k1 = z / (np.pi * f)
        self.k2 = 1 / ((2 * np.pi * f) * (2 * np.pi * f))
        self.k3 = r * z / (2 * np.pi * f)
        # self.x = x0
        print(f"Setup system with k1={self.k1}, k2={self.k2}, k3={self.k3}")
        
        self.x_prev = x0
        self.dx = 0
        self.dx_prev = 0

        self.y = 0
        self.y_prev = 0
        self.dy = 0
        self.dy_prev = 0

        self.dt = dt
        # print(f"{self.dx=}")
        # print(f"{self.x_prev=}")



    def step(self, x):
        # print(f"{x=}")
        self.dx = x - self.x_prev
        # print(f"{self.dx=}")
        self.y = self.y_prev + self.dt * self.dy_prev
        # print(f"{self.y=} {self.y_prev=}")
        self.dy = self.dy_prev + self.dt * (x + self.k3 * self.dx - self.y - self.k1*self.dy_prev)/self.k2
        # print(f"{self.dy=} {self.dy_prev=}")

        self.x_prev = x
        self.dx_prev = self.dx
        self.y_prev = self.y
        self.dy_prev = self.dy
        return self.y

    #x is the reset value
    def reset(self, x0):
        self.x_prev = x0
        self.dx = 0
        self.dx_prev = 0

        self.y = x0
        self.y_prev = x0
        self.dy = 0
        self.dy_prev = 0
