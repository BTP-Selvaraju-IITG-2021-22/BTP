#!/bin/python3

import pulp
from typing import List
import numpy as np
import scipy.optimize

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

class Network:
    
    def __init__(self, serviceRates : List[float], players : List[Player]) -> None:
        self.players : list[Player] = []
        self.state = []
        self.numNodes = len(serviceRates)
        self.serviceRates = serviceRates
        
        for player in players:
            self.__add_player(player)
        
        self.C : List[List[List[float]]] = self.__calcC()
    
    def __add_player(self, player : Player) -> None:
        if len(self.players) > 0:
            assert(player.reqNodes == len(self.serviceRates))
        self.players.append(player)
        
        # Fix: possible that initial state is not feasible
        initState = [0]*player.nChoices
        initState[0] = 1
        self.state.append(initState)
        
    def __calcC(self) -> List[List[List[float]]]:
        C = [[[0.0 for i in range(self.numNodes)]
               for r in range(self.players[k].nChoices)]
                for k in range(len(self.players))]
        
        for k in range(len(self.players)):
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
            for k in range(len(self.players)):
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
        for j in range(len(self.players)):
            sm = 0
            for p in strategies[j]:
                sm += p
                if not (p >= 0): return False
            if not (sm == 1): return False
        
        # total arrival rate at node is less than service rate                        
        for i in range(self.numNodes):
            arrv = 0
            for k in range(len(self.players)):
                for r in range(self.players[k].nChoices):
                    arrv += strategies[k][r]*self.C[k][r][i]
            if not (arrv <= self.serviceRates[i]): return False
        
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
            for k in range(len(self.players)):
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
        for j in range(curPlayer.nChoices):
            A_ineq[self.numNodes + 1 + j, j] = 1.0
            ub[self.numNodes + 1 + j] = 1
            lb[self.numNodes + 1 + j] = 0
        
        linearConstraints = scipy.optimize.LinearConstraint(A_ineq, lb, ub, keep_feasible=keep_feasible)
        costFn = lambda p : self.waitTime(playerId, strategy=p)
        
        p0 = None
        satisfies = False
        while not satisfies:
            p0 = np.random.uniform(size=(curPlayer.nChoices,)) # initial guess for `curPlayer`
            p0 = p0/np.sum(p0)
            Ap0 = A_ineq.dot(p0)
            satisfies = np.all(lb <= Ap0) and np.all(Ap0 <= ub)
            # print("checking p0 = ", p0, "... satisfies?" , satisfies)
        
        optRes = scipy.optimize.minimize(costFn, p0, constraints=linearConstraints)
        return optRes.x
    
    
    def calcPureNashEquilibrium(self):
        
        # find a satisfiable strategy
        while not self.checkStrategies(self.state):
            self.state = [np.random.uniform(size=player.nChoices) for player in self.players]
            for i in range(len(self.state)):
                self.state[i] = self.state[i]/np.sum(self.state[i])
        
        
        nIter = 5
        eps = 0.0001
        for iter in range(nIter):
            print(f"Nash Iteration {iter}")
            anyNew = False
            print("current Strats : ", self.state)
            for j in range(len(self.players)):
                oldStrat = self.state[j]
                self.state[j] = self.bestResponse(j)
                print(f"new strat for player {j}", self.state[j])
                if np.max(np.abs(oldStrat - self.state[j])) > eps:
                    anyNew = True
            if not anyNew:
                print(f"Nothing new...")
        
        return self.state
                