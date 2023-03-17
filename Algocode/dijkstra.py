# 图
graph = {}
graph['start'] = {}
graph['start']['a'] = 6
graph['start']['b'] = 2
graph['a'] = {}
graph['a']['final'] = 1
graph['b'] = {}
graph['b']['a'] = 3
graph['b']['final'] = 5

# 开销
infinity =  float('inf')
costs = {}
costs['a'] = 6
costs['b'] = 2
costs['final'] = infinity

# 父节点
parents = {}
parents['a'] = 'start'
parents['b'] = 'start'
parents['final'] = None

# 已处理点
processed = []
def find_lowest_cost_node(costs):
    lowest_cost = float('inf')
    lowest_cost_node = None
    for node in costs.keys():
        cost = costs[node]
        if cost < lowest_cost and node not in processed:
            lowest_cost_node = node
            lowest_cost = cost
    return lowest_cost_node

node = find_lowest_cost_node(costs)
while node is not None:
    cost = costs[node]
    neighbors = graph[node]
    for n in neighbors.keys():
        new_cost = cost + neighbors[n]
        if costs[n] > new_cost:
            costs[n] = new_cost
            parents[n] = node
    processed.append(node)
    node = find_lowest_cost_node(costs)