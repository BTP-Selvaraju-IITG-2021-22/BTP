from non_cooperative import Route, Player, Network

r1 = Route([0.3, 0, 0], [
    [0, 1.0, 0],
    [0, 0.4, 0.6],
    [0, 0, 0]
])

p1 = Player([r1], 0)
p2 = Player([r1], 1)

network = Network([2.0, 2.0, 2.0], [p1, p2])
print(network.waitTime(0))
network.bestResponse(0)