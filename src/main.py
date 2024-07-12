from core.backend.net import MLP
from core.backend.functions import Sigmoid,Mse

mse = Mse()
sigmoid = Sigmoid()
mlp = MLP(4,[5,5,2], sigmoid)

inputs = [2.0, 3.0, -1.0, 0.5]
targets = [1.0,0.0]

lr = 0.1

for _ in range(100):

    output = mlp(inputs)

    loss = mse(output, targets)

    loss_derivative = mse.derivative(output,targets)

    mlp.backwards(loss_derivative)
    mlp.update(lr)

    if _ % 10 == 0 or _ == 1:
        print(output)
        print(targets)
        print(f'loss: {loss}')