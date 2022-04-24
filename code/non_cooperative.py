#!/bin/python3

import pulp
from typing import List
import numpy as np

class Route:
    def __init__(self,gamma, R) -> None:
        self.gamma = gamma
        self.R = np.array(R)
        self.Q = np.linalg.inv(np.identity(self.R.shape[0]) - self.R)

class Player:
    counter = 0
    
    def __init__(self, routes : List[Route], id=None):
        
        self.id = id
        Player.counter += 1
        
        gammas = [x.gamma for x in routes]
        Rs = [x.R for x in routes]
        
        assert(len(gammas) == len(Rs))
        self.reqNodes = len(gammas[0])
        self.nChoices = len(gammas)
        self.totalArrival = sum(gammas[0])
        
        for gamma, R in zip(gammas, Rs):
            assert(len(gamma) == len(R) == self.reqNodes)
            assert(sum(gamma) == self.totalArrival)
            
        self.routingChoices = routes
    
    def genVars(self):
        assert(self.id != None)
        vars = []
        for i in range(self.nChoices):
            vars.append(pulp.LpVariable(f"p_{self.id}_{i}", lowBound=0))
        return vars
    

class Network:
    
    def __init__(self, serviceRates : List[float], players : List[Player]) -> None:
        self.players : list[Player] = []
        self.state = []
        self.numNodes = len(serviceRates)
        self.serviceRates = serviceRates
        
        for player in players:
            self.__add_player(player)
    
    def __add_player(self, player : Player) -> None:
        if len(self.players) > 0:
            assert(player.reqNodes == len(self.serviceRates))
        self.players.append(player)
        
        # Fix: possible that initial state is not feasible
        initState = [0]*player.nChoices
        initState[0] = 1
        self.state.append(initState)
    
    def waitTime(self, playerId : int, strategies=None):
        if strategies == None:
            strategies = self.state
        
        curPlayer = self.players[playerId]
            
        C = [[[0 for i in range(self.numNodes)]
               for r in range(self.players[k].nChoices)]
                for k in range(len(self.players))]
        
        for k in range(len(self.players)):
            for r in range(self.players[k].nChoices):
                for i in range(self.numNodes):
                    for j in range(self.numNodes):
                        route_r = self.players[k].routingChoices[r]
                        C[k][r][i] += route_r.gamma[j] * route_r.Q[j,i]

        waits = []
        # waits[i] = $\frac{1}{\mu_i - \sum_{k}\sum_{r} p^{(k,r)} C^{(k,r,i)}}$
        for i in range(self.numNodes):
            arrv = 0
            for k in range(len(self.players)):
                for r in range(self.players[k].nChoices):
                    arrv += strategies[k][r]*C[k][r][i]
            waits.append(1.0/(self.serviceRates[i] - arrv))
        
        l = playerId
        waitTime = 0
        for m in range(curPlayer.nChoices):
            temp = 0
            for i in range(self.numNodes):
                temp += C[l][m][i]*waits[i]
            waitTime += (strategies[l][m]/curPlayer.totalArrival)*temp
        
        return waitTime

    def genFeasibleSet(self):
        condn = []
        
        # 1. sum of probabilities is 1
        # 2. probabilities are >= 0
        for j in range(len(self.players)):
            sm = 0
            for p in self.state[j]:
                sm += p
                condn.append((p >= 0, ""))
            condn.append((sm == 1, f"sum of probab of player {j}"))
        
        # total arrival rate at node is less than service rate
        C = [[[0 for i in range(self.numNodes)]
               for r in range(self.players[k].nChoices)]
                for k in range(len(self.players))]
        
        for k in range(len(self.players)):
            for r in range(self.players[k].nChoices):
                for i in range(self.numNodes):
                    for j in range(self.numNodes):
                        route_r = self.players[k].routingChoices[r]
                        C[k][r][i] += route_r.gamma[j] * route_r.Q[j,i]
                        
        for i in range(self.numNodes):
            arrv = 0
            for k in range(len(self.players)):
                for r in range(self.players[k].nChoices):
                    arrv += self.state[k][r]*C[k][r][i]
            condn.append((arrv <= self.serviceRates[i], f"service rate at node {i}"))
        
        return [x for x in condn if type(x[0]) != type(True)]
        

    def bestResponse(self, playerId : int):
        # set state to the best response of player `playerId`
        # Changes will only be made of state[playerId]
        self.state[playerId] = self.players[playerId].genVars()
        prob = pulp.LpProblem(f"bestResponseForPlayer{playerId}", pulp.const.LpMinimize)
        prob += self.waitTime(playerId), f"wait time for player {playerId}"
        for x in self.genFeasibleSet():
            prob += x
        print(prob)
    