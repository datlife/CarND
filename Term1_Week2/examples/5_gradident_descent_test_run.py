from Week2.nnPy import f, df, gradient_descent_update
import random

x = random.randint(0, 10000)
learning_rate = 0.01
epochs = 1000

for i in range(epochs + 1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}:  Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)

