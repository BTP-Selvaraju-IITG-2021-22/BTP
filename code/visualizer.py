#!/bin/python3
import json
import graphviz

filename = 'qnet1.json'
dot = graphviz.Digraph(comment=filename)
dot.attr('node', shape='circle', fixedsize='true', width='0.9')


dot.node('input',style='dotted')
with open(filename) as filePtr:
    data = json.load(filePtr)
    for node in data['Nodes']:
        dot.node(node['name'], node['name'] + u'\nμ=' + str(node['srvRate']))
        if node['extRate'] > 0:
            dot.edge('input', node['name'], label=u'λ='+str(node['extRate']))
        for outNode in node['out']:
            dot.edge(node['name'], outNode['node'],
                    label=str(outNode['probab']))

    dot.view()
