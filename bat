pip install numpy matplotlib
import numpy as np
import matplotlib.pyplot as plt

# تعریف تابع Rastrigin
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

# تعریف الگوریتم خفاش
class BatAlgorithm:
    def __init__(self, func, dim, n_bats=30, max_iter=1000, A=0.5, r=0.5, f_min=0, f_max=2):
        self.func = func
        self.dim = dim
        self.n_bats = n_bats
        self.max_iter = max_iter
        self.A = A
        self.r = r
        self.f_min = f_min
        self.f_max = f_max
        self.lb = -5.12
        self.ub = 5.12
        self.bats = np.random.uniform(self.lb, self.ub, (self.n_bats, self.dim))
        self.velocities = np.zeros((self.n_bats, self.dim))
        self.frequencies = np.zeros(self.n_bats)
        self.loudness = np.full(self.n_bats, self.A)
        self.pulse_rate = np.full(self.n_bats, self.r)
        self.best_bat = None
        self.best_score = float('inf')

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.n_bats):
                self.frequencies[i] = self.f_min + (self.f_max - self.f_min) * np.random.rand()
                self.velocities[i] += (self.bats[i] - self.best_bat) * self.frequencies[i]
                candidate_solution = self.bats[i] + self.velocities[i]
                candidate_solution = np.clip(candidate_solution, self.lb, self.ub)
                if np.random.rand() > self.pulse_rate[i]:
                    candidate_solution = self.best_bat + 0.001 * np.random.randn(self.dim)
                candidate_score = self.func(candidate_solution)
                if candidate_score < self.best_score and np.random.rand() < self.loudness[i]:
                    self.bats[i] = candidate_solution
                    self.loudness[i] *= 0.9
                    self.pulse_rate[i] *= 0.9
                    if candidate_score < self.best_score:
                        self.best_bat = candidate_solution
                        self.best_score = candidate_score
            print(f"Iteration {t+1}, best score: {self.best_score}")
        return self.best_bat, self.best_score

# پارامترهای الگوریتم
dim = 10
n_bats = 30
max_iter = 1000

# ایجاد شیء الگوریتم خفاش و اجرای بهینه‌سازی
ba = BatAlgorithm(rastrigin, dim, n_bats, max_iter)
best_solution, best_score = ba.optimize()

print("Best Solution:", best_solution)
print("Best Score:", best_score)
