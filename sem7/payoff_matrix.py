#!/bin/bash
import numpy as np
from itertools import product

def read_floats():
    inp = input()
    inp = list(map(float, filter(lambda x: x != '', inp.split(' '))))
    if len(inp) == 1:
        return inp[0]
    else:
        return tuple(inp)


### user inputs
N, C = map(int, read_floats())
service_rate = read_floats()

arrv_rate = [0]*N
routes = [[] for i in range(N)]

for j in range(N):
    num_routes, arr = read_floats()
    num_routes = int(num_routes)
    arrv_rate[j] = arr
    for _ in range(num_routes):
        route = list(map(lambda x: int(x)-1, read_floats()))[1:] #convert to 0 indexed int and remove the first element
        routes[j].append(route)

### end user inputs

idx = 0
def calc_payoffs(S):
    global idx
    idx += 1
    
    
    freq_node = [0]*C

    for j in range(N):
        for i in range(len(routes[j])):
            for node in routes[j][i]:
                freq_node[node] += S[j][i]*arrv_rate[j]

    sojourn_node = [0]*C
    for i in range(C):
        assert(service_rate[i] > freq_node[i])
        sojourn_node[i] = 1/(service_rate[i] - freq_node[i])
    
    expected_waiting_time = [0]*N
    
    for j in range(N):
        for i in range(len(routes[j])):
            route_wait = 0
            for node in routes[j][i]:
                route_wait += sojourn_node[node]
            expected_waiting_time[j] += route_wait*S[j][i]
    
    
    # print(f'For (idx={idx}) S={S}: waiting times = {expected_waiting_time}')
    
    return expected_waiting_time

## discrete payoff matrix





payoff = np.zeros(tuple(list(len(routes[j]) for j in range(N)) + [N]))
indices = product(*tuple(range(len(routes[j])) for j in range(N)))
for index in indices:
    # index = (1,4,2,0,...) -> corresponding to choices of route
    S = [[0]*len(routes[j]) for j in range(N)]
    for j, choice in enumerate(index):
        S[j][choice] = 1
    payoff[index] = calc_payoffs(S)

print(payoff)