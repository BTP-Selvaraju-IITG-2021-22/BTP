#!/bin/python3
import json
import graphviz
from non_cooperative import Route

service = [1.2, 1.2, 2.1]
route = Route([1, 0], [
    [0.1, 0.3],
    [0, 0],
])

dot = graphviz.Digraph()
dot.attr('node', shape='circle', fixedsize='true', width='1.9')

for i in range(len(route.gamma)):
    label = f'μ={service[i]}'
    if route.gamma[i] > 0:
        label += f'\nγ={route.gamma[i]}'
    if sum(route.R[i]) < 1:
        label += f'\np0={1-sum(route.R[i]):.2f}'
    dot.node(f'{i}', label=label)

for i in range(len(route.gamma)):
    for j in range(len(route.gamma)):
        if route.R[i][j] > 0:
            dot.edge(str(i), str(j), label=f'{route.R[i][j]}')

# dot.node('input',style='dotted')
# with open(filename) as filePtr:
#     data = json.load(filePtr)
#     for node in data['Nodes']:
#         dot.node(node['name'], node['name'] + u'\nμ=' + str(node['srvRate']))
#         if node['extRate'] > 0:
#             dot.edge('input', node['name'], label=u'λ='+str(node['extRate']))
#         for outNode in node['out']:
#             dot.edge(node['name'], outNode['node'],
#                     label=str(outNode['probab']))

dot.view()
