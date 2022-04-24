from cProfile import label
from non_cooperative import Route, Player, Network
import matplotlib.pyplot as plt

r11 = Route([0, 4], [
    [0, 0],
    [0, 0],
])

r12 = Route([4, 0], [
    [0, 0],
    [0, 0],
])

r21 = Route([0, 3], [
    [0, 0],
    [0, 0],
])

r22 = Route([3, 0], [
    [0, 0],
    [0, 0],
])


p0 = Player([r11, r12])
p1 = Player([r21, r22])

network = Network([20., 10.0], [p0, p1])
# print(network.waitTime(0))
# print(network.calcPureNashEquilibrium())
# print(network.correlatedEquilbrium())
# print(network.calcDiscreteCosts())

mwew = network.multiplicativeWeights(eta=0.5, nSteps=50)
for i in range(len(mwew[0])):
    plt.plot([x[i] for x in mwew], '.-', label=f'Player {i}')
plt.xlabel('num iterations')
plt.ylabel('expected waiting time')
plt.legend()
plt.show()