## Input format

The first line contains two numbers: the number of players $N$ and number of nodes $C$.

The next line contains $C$ integers denoting the service rates of nodes.

Then for each player, the first line contains: number of routes $|R^{(j)}|$ and arrival rate $\lambda^{(j)}$

Then then next $|R^{(j)}|$ lines contains: an integer $k$ followed by $k$ integers $r_1 ~ , ~ r_2 ~ \ldots, ~ r_k$ which denotes a route
$(r_1 ~ , ~ \ldots ~ , ~ r_k)$

```
N C
    Rj  Lj
        k  r1  r2 ... rk
```