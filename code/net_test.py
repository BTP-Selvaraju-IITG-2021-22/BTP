from non_cooperative import Route, Player, Network

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


p0 = Player([r11, r12], 0)
p1 = Player([r21, r22], 1)

network = Network([20., 20.0], [p0, p1])
print(network.waitTime(0))
# print(network.calcPureNashEquilibrium())
network.correlatedEquilbrium()
# print(network.calcDiscreteCosts())