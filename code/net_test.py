from non_cooperative import Route, Player, Network

r1 = Route([0.3, 0, 0], [
    [0, 1.0, 0],
    [0, 0.4, 0.6],
    [0, 0, 0]
])
r2 = Route([0.3, 0, 0], [
    [0, 0.3, 0.6],
    [0, 0.4, 0.6],
    [0.1, 0, 0]
])

r3 = Route([0.1, 0.1, 0.1], [
    [0, 0.3, 0.6],
    [0, 0.4, 0.6],
    [0.1, 0, 0]
])

r4 = Route([0.1, 0.1, 0.1], [
    [0.3, 0.6, 0],
    [0, 0, 1.0],
    [0.0, 0, 0]
])


p0 = Player([r1, r2], 0)
p1 = Player([r3, r4], 1)

network = Network([2.0, 2.0, 2.0], [p0, p1])
print(network.waitTime(0))
print(network.calcPureNashEquilibrium())