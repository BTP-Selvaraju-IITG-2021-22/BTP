## Format for open jackson queueing network
```json
Node {
    "name" : string,
    "srvRate" : double,
    "extRate" : double,
    "out" : [
        {"node" : string, "probab" : double}
        // sum of probablities must be <= 1 (if < 1 remaining probablity is of leaving the system)
    ]
}
```
<!-- NodeName, Service Rate, External arrival rate, outgoing edge 1, probab 1, [outgoing edge 2, probab 2, [...]] -->

## Visualizer of open jackson network using graphviz

## Program for waiting time in open jackson network

## Implement best response algorithm