#!/bin/python3

from typing import List
from xml.etree.ElementInclude import include
import numpy as np
import scipy.optimize
import itertools
import pulp
import math
import graphviz

class Route:
    def __init__(self,gamma, R) -> None:
        self.gamma = gamma
        self.R = np.array(R)
        self.Q = np.linalg.inv(np.identity(self.R.shape[0]) - self.R)

class Player:
    counter = 0
    
    def __init__(self, routes : List[Route]):
        
        gammas = [x.gamma for x in routes]
        Rs = [x.R for x in routes]
        
        assert(len(gammas) == len(Rs))
        self.reqNodes = len(gammas[0])
        self.nChoices = len(gammas)
        self.totalArrival = sum(gammas[0])
        
        for gamma, R in zip(gammas, Rs):
            assert(len(gamma) == len(R) == self.reqNodes)
            assert(abs(sum(gamma) - self.totalArrival) < 1e-4)
            
        self.routingChoices = routes

class Network:
    
    def __init__(self, serviceRates : List[float], players : List[Player]) -> None:
        self.players : list[Player] = []
        self.state = []
        self.numNodes = len(serviceRates)
        self.serviceRates = serviceRates
        self.nPlayers = len(players)
        
        for player in players:
            self.__add_player(player)
        
        self.C : List[List[List[float]]] = self.__calcC()
    
    def __add_player(self, player : Player) -> None:
        if self.nPlayers > 0:
            assert(player.reqNodes == len(self.serviceRates))
        self.players.append(player)
        
        # Fix: possible that initial state is not feasible
        initState = [0]*player.nChoices
        initState[0] = 1
        self.state.append(initState)
        
    def __calcC(self) -> List[List[List[float]]]:
        C = [[[0.0 for i in range(self.numNodes)]
               for r in range(self.players[k].nChoices)]
                for k in range(self.nPlayers)]
        
        for k in range(self.nPlayers):
            for r in range(self.players[k].nChoices):
                for i in range(self.numNodes):
                    for j in range(self.numNodes):
                        route_r = self.players[k].routingChoices[r]
                        C[k][r][i] += route_r.gamma[j] * route_r.Q[j,i]
        return C
       
    def waitTime(self, playerId : int, strategy=None):
        strategies = self.state
        if strategy is not None:
            strategies[playerId] = strategy
        
        curPlayer = self.players[playerId]

        waits = []
        # waits[i] = $\frac{1}{\mu_i - \sum_{k}\sum_{r} p^{(k,r)} C^{(k,r,i)}}$
        for i in range(self.numNodes):
            arrv = 0
            for k in range(self.nPlayers):
                for r in range(self.players[k].nChoices):
                    arrv += strategies[k][r]*self.C[k][r][i]
            waits.append(1.0/(self.serviceRates[i] - arrv))
        
        l = playerId
        waitTime = 0
        for m in range(curPlayer.nChoices):
            temp = 0
            for i in range(self.numNodes):
                temp += self.C[l][m][i]*waits[i]
            waitTime += (strategies[l][m]/curPlayer.totalArrival)*temp
        
        return waitTime

    def checkStrategies(self, strategies):
        # 1. sum of probabilities is 1
        # 2. probabilities are >= 0
        for j in range(self.nPlayers):
            sm = 0
            for p in strategies[j]:
                sm += p
                if not (p >= 0): return False
            if not (sm == 1): return False
        
        # total arrival rate at node is less than service rate                        
        for i in range(self.numNodes):
            arrv = 0
            for k in range(self.nPlayers):
                for r in range(self.players[k].nChoices):
                    arrv += strategies[k][r]*self.C[k][r][i]
            if not (arrv < self.serviceRates[i]): return False
        
        return True
        
    def bestResponse(self, playerId : int):
        # set state to the best response of player `playerId`
        # Changes will only be made of state[playerId]
        curPlayer = self.players[playerId]
        
        linearConstraints = []
        A_ineq = np.zeros((self.numNodes, curPlayer.nChoices))
        lb_ineq = np.zeros((self.numNodes, ))
        ub_ineq = np.zeros((self.numNodes, ))
        keep_feasible = [True]*(self.numNodes)
        
        # total arrival rate  < service rate for node i
        for i in range(self.numNodes):
            for j in range(curPlayer.nChoices):
                A_ineq[i,j] = self.C[playerId][j][i]
            lb_ineq[i] = 0
            muhat = self.serviceRates[i]
            for k in range(self.nPlayers):
                for r in range(self.players[k].nChoices):
                    muhat = muhat - self.state[k][r]*self.C[k][r][i]
            ub_ineq[i] = muhat
        linearConstraints.append(scipy.optimize.LinearConstraint(A_ineq, lb_ineq, ub_ineq))
        
        
        # sum of probabilities is 1
        A_eq = np.zeros((1, curPlayer.nChoices)) + 1
        lb_eq = np.zeros((1,)) + 1
        ub_eq = lb_eq
        linearConstraints.append(scipy.optimize.LinearConstraint(A_eq, lb_eq, ub_eq))
        
        # 1 >= each probabilities >= 0
        bounds = scipy.optimize.Bounds(np.zeros(curPlayer.nChoices,), 
                                       1 + np.zeros(curPlayer.nChoices,))
        
        costFn = lambda p : self.waitTime(playerId, strategy=p)
        
        p0 = None
        satisfies = False
        while not satisfies:
            p0 = np.random.uniform(size=(curPlayer.nChoices,)) # initial guess for `curPlayer`
            p0 = p0/np.sum(p0)
            Ap0 = A_ineq.dot(p0)
            satisfies = np.all(lb_ineq <= Ap0) and np.all(Ap0 <= ub_ineq)
        
        optRes = scipy.optimize.minimize(costFn, p0, constraints=linearConstraints, bounds=bounds)
        return optRes.x
    
    def calcPureNashEquilibrium(self):
        
        # find a satisfiable strategy
        while not self.checkStrategies(self.state):
            self.state = [np.random.uniform(size=player.nChoices) for player in self.players]
            for i in range(len(self.state)):
                self.state[i] = self.state[i]/np.sum(self.state[i])
        
        maxIter = 50
        eps = 0.001
        for iter in range(maxIter):
            anyNew = False
            for j in range(self.nPlayers):
                oldStrat = self.state[j]
                self.state[j] = self.bestResponse(j)
                if np.max(np.abs(oldStrat - self.state[j])) > eps:
                    anyNew = True
            if not anyNew:
                break
        
        socialCost = 0
        for i in range(self.nPlayers):
            socialCost += self.waitTime(i)
        
        return self.state, socialCost
    
    def calcDiscreteCosts(self, includeNonFeasible=False):
        strategySpace = itertools.product(*[list(range(pl.nChoices)) for pl in self.players])
        cost = {}
        
        for strategy in strategySpace:
            curCost = np.zeros(self.nPlayers)
            for j, rj in enumerate(strategy):
                self.state[j] = np.zeros((self.players[j].nChoices,))
                self.state[j][rj] = 1.0
            
            if not self.checkStrategies(self.state):
                if includeNonFeasible:
                    curCost = np.infty + curCost
                else:
                    continue
            else:
                for j in range(self.nPlayers):
                    curCost[j] = self.waitTime(j)
            
            cost[strategy] = tuple(curCost.tolist())
        
        return cost
    
    def correlatedEquilbrium(self):
        cost = self.calcDiscreteCosts(includeNonFeasible=True)
        
        # replace infty with very large number
        M = 1000.0
        for s in cost:
            if cost[s][0] == np.infty:
                cost[s] = tuple([M]*self.nPlayers)
        
        p = {}
        systemLoad = 0
        
        prob = pulp.LpProblem("correlatedEq", sense=pulp.const.LpMinimize)
        totalProb = 0
        for s in cost:
            p[s] = pulp.LpVariable(f'p_{s}',lowBound=0,upBound=1)
            totalProb += p[s]
            systemLoad += p[s] * sum(cost[s])
            
        prob += systemLoad, "average wait time of players"
        prob += totalProb == 1
        
        def edit_tuple(tup, ind, x):
            return tup[:ind] + (x,) + tup[ind+1:]
        
        
        playerChoices = [list(range(pl.nChoices)) for pl in self.players]
        for i, curPlayer in enumerate(self.players):
            for si, sip in itertools.product(range(curPlayer.nChoices), repeat=2):
                if si == sip : continue
                playerChoices[i] = [si]
                
                coordCost = 0
                ignoreCost = 0
                
                for s in itertools.product(*playerChoices):
                    s1 = edit_tuple(s, i, sip)
                    coordCost += cost[s][i]*p[s]
                    ignoreCost += cost[s1][i]*p[s]
                    
                prob += coordCost <= ignoreCost, f"player {i} -- {si} better than {sip}"
            playerChoices[i] = list(range(curPlayer.nChoices))
        
        prob.solve()
        
        res = {}
        for s in p:
            v = p[s]
            res[s] = v.varValue

        return res, prob.objective.value()
    
    def discreteCooperativeOptimal(self):
        return min(map(sum, self.calcDiscreteCosts().values()))
    
    def multiplicativeWeights(self, eta=0.2, nSteps=5):
        costs = self.calcDiscreteCosts(includeNonFeasible=True)
        states = list(costs.keys())
        
        p = [[[1.0/pl.nChoices]*pl.nChoices for pl in self.players] for _ in range(nSteps+1)]
        loss = [[[0]*pl.nChoices for pl in self.players] for _ in range(nSteps+1)]
        EW = [[0]*self.nPlayers for _ in range(nSteps)]
        
        for t in range(nSteps):
            for j in range(self.nPlayers):
                for r in range(self.players[j].nChoices):
                    loss[t][j][r] += sum(
                        costs[s][j]*math.prod(p[t][k][s[k]] for k in range(self.nPlayers) if k != j) 
                        for s in states if s[j] == r
                    )
                    p[t+1][j][r] = p[t][j][r]*math.pow(1-eta, loss[t][j][r])
                
                # normalize p[t+1][j]
                sm = sum(p[t+1][j])
                for r in range(self.players[j].nChoices):
                    p[t+1][j][r] /= sm

                
                EW[t][j] = sum(loss[t][j][r]*p[t][j][r] for r in range(self.players[j].nChoices))
        
        return EW
    
    def visualize(self) -> graphviz.Digraph:
        g = graphviz.Digraph()
        g.attr('node', shape='circle', fixedsize='true', width='2.0')
        
        for k, player in enumerate(self.players):
            for r, route in enumerate(player.routingChoices):
                name = f'p{k}_{r}'
                with g.subgraph(name='cluster_'+name) as c:
                    c.attr(label=f'Player {k}, routeChoice {r}')
                    c.attr(color='blue')

                    for i in range(len(route.gamma)):
                        label = f'μ={self.serviceRates[i]}'
                        if route.gamma[i] > 0:
                            label += f'\nγ={route.gamma[i]}'
                        if sum(route.R[i]) < 1:
                            label += f'\np0={1-sum(route.R[i]):.2f}'
                        c.node(f'{name}_{i}', label=label)

                    for i in range(len(route.gamma)):
                        for j in range(len(route.gamma)):
                            if route.R[i][j] > 0:
                                c.edge(f'{name}_{i}', f'{name}_{j}', label=f'{route.R[i][j]}')
        
        return g