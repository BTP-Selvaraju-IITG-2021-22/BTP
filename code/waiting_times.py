import json
import numpy as np

def jacksonParams(raw_data):
    n = len(raw_data['Nodes'])
    gammas = []
    Rs = []
    mu = np.zeros((n,))
    
    for i, node in enumerate(raw_data['Nodes']):
        mu[i] = node['srvRate']
    
    for i, cust in enumerate(raw_data['Customer']):
        gammas.append(np.array(cust['arrival'], dtype=float))
        Rmat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                Rmat[i,j] = float(cust['routing'][i][j])
        
        Rs.append(Rmat)
    
    print('R=',Rs)
    print('mu=',mu)
    print('gamma=',gammas)
    return mu, gammas, Rs

def networkWait(mu, gammas, Rs):
    N = len(mu) # number of nodes
    M = len(Rs) # number of classes
    lambdas = [None]*M
    
    # make sure in the right format
    mu = np.reshape(mu, (1,N))      # row vector
    for i in range(M):
        gammas[i] = np.reshape(gammas[i], (1,N))        # row vector
        # Rs[i] = np.reshape(Rs[i], (N,N))
    
    pinit = gammas[i]/np.sum(gammas)
    
    netLambda = np.zeros((1,N))     # row vector
    for i in range(M):
        IRI = np.linalg.inv(np.identity(N) - Rs[i])
        # print('gammas[i] = ', gammas[i])
        # print('IRI = ', IRI)
        lambdas[i] = np.dot(gammas[i], IRI)
        netLambda += lambdas[i]
        
    # print('lambdas = ', lambdas)
    
    EW = 1.0/(mu - netLambda)
    
    waitTime = [0]*M
    
    for i in range(M):
        IRI = np.linalg.inv(np.identity(N) - Rs[i])
        lambdas[i] = np.dot(gammas[i], IRI)
        EY = np.dot(pinit, IRI)
        waitTime[i] = np.dot(EY, np.transpose(EW))

    print(waitTime)
    return waitTime

filename = 'qnet1.json'
with open(filename) as filePtr:
    graph = json.load(filePtr)
    mu, gammas, Rs = jacksonParams(graph)
    networkWait(mu, gammas, Rs)