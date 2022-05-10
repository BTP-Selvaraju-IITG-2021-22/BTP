from cProfile import label
from non_cooperative import Route, Player, Network
import matplotlib.pyplot as plt
from random import random
import graphviz

baseRoute = [
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1],
]
baseGamma = [1, 0, 0, 1, 0, 0]
outBase =   [0, 0, 0, 0, 1, 1]

def genRandomRoute(totArrv):
    N = len(baseRoute)
    route = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            route[i][j] = random()*baseRoute[i][j]
        p0 = outBase[i]*random()
        norm = p0 + sum(route[i])
        route[i] = [x/norm for x in route[i]]
    
    gammas = [x*random() for x in baseGamma]
    norm = sum(gammas)
    gammas = [totArrv*x/norm for x in gammas]
    
    print(sum(gammas))
    # print(route)
    # print('\n\n')
    return Route(gammas, route)

def tostr_route(route):
    res = '\\begin{pmatrix}'
    for i in route.R:
        res += ' & '.join([f'{x:.2f}' for x in i])
        res += '\\\\\n'
    res += '\\end{pmatrix}'
    
    res += '\n\n'
    res += '\\left[' + ' '.join([f'{x:.2f}' for x in route.gamma]) + ' \\right]'
    return res + '\n'

# b/w 20 and 30
serviceRates = [random()*10 + 50 for _ in range(len(baseRoute))]

arrv = [5.0, 8.0]
nChoices = 2

players = []
for i in range(len(arrv)):
    players.append(Player([genRandomRoute(arrv[i]) for _ in range(nChoices)]))

network = Network(serviceRates, players)

# network = Network([20., 10.0], [p0, p1])
# print(network.waitTime(0))
pureNash, pureSoc = network.calcPureNashEquilibrium()
print("pure Nash social cost = ", pureSoc)
with open('output/pure.txt', 'w') as f:
    f.write(str(pureNash)+ '\n')
    f.write(str(pureSoc))

corr, corrSoc = network.correlatedEquilbrium()
print("correlated social Cost = " ,corrSoc)
with open('output/corr.txt', 'w') as f:
    f.write(str(corr))
    f.write(str(corrSoc))

# print(network.discreteCooperativeOptimal())

mwew = network.multiplicativeWeights(eta=0.3, nSteps=500)
for i in range(len(mwew[0])):
    plt.plot([x[i] for x in mwew], '.-', label=f'Player {i}')
plt.xlabel('Iterations')
plt.ylabel('Expected waiting time')
plt.legend()
plt.savefig('output/mw.png')

plt.clf()
plt.plot([sum(x) for x in mwew], '.-')
plt.xlabel('Iterations')
plt.ylabel('Social Cost')
plt.savefig('output/mw_social.png')



pli = 0 # player
rti = 0 # route 
ndpos = [(1,2),(2,3),(3.5,3),(2,1),(3.5,1),(4.5,2)]
scale=3
for pli in range(network.nPlayers):
    for rti in range(network.players[pli].nChoices):
        
        
        g = graphviz.Digraph('G', engine="neato", filename='ex.gv',format='pdf')
        g.attr(size='30')
        g.attr('node', shape='circle', fixedsize='true', width='2')
        route = network.players[pli].routingChoices[rti]

        for i in range(6):
            label = f'node{i}'
            label += f'\nμ={network.serviceRates[i]:.2f}'
            if route.gamma[i] > 0.001:
                label += f'\nγ={route.gamma[i]:.2f}'
            if 1-sum(route.R[i]) > 0.001:
                label += f'\np0={1.-sum(route.R[i]):.2f}'
            g.node(str(i), label=label, pos=f'{ndpos[i][0]*scale},{ndpos[i][1]*scale}!')


        for i in range(len(route.gamma)):
            for j in range(len(route.gamma)):
                if route.R[i][j] > 1e-2:
                    g.edge(str(i),str(j),label=f'{route.R[i][j]:.2f}')

        g.render(f'output/p{pli}_r{rti}',format='png')
        with open(f'output/p{pli}_r{rti}.txt', 'w') as f:
            f.write(tostr_route(route))

        
# g.node('0',pos='1,2!')
# g.node('1',pos='2,3!')
# g.node('2',pos='3,3!')
# g.node('3',pos='2,1!')
# g.node('4',pos='3,1!')
# g.node('5',pos='4,2!')

print(network.calcDiscreteCosts())