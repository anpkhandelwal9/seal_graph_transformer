
cimport cython

from libcpp.vector cimport vector as cpp_vector
from libcpp.queue cimport queue as cpp_queue

cpdef bfs(edge_index, num_nodes):
    num_edges = edge_index.shape[1]
    cdef cpp_vector[int]* adj_list[num_nodes]
    for i in range(num_egdes):
        
        if adj_list[edge_index[0][i]] == NULL:
            cdef cpp_vector[int] temp_vector;
            adj_list[edge_index[0][i]] = temp_vector;
        adj_list[edge_index[0][i]].push_back(edge_index[0][j])
    
    cdef bool visited[num_nodes]
    cdef cpp_queue[int] bfs_queue
    cdef int levels[num_nodes]
    for i in range(num_nodes):
        levels[i] = limit
        visited[i] = False
    queue.push(node)

    while not(queue.empty()):          # Creating loop to visit each node
        m = queue.pop()
        if levels[m] < limit:
            for neighbour in range(adj_list[m].size()):
                if not(visited[neighbour]):
                    visited[node_id] = True
                    queue.push(node_id)
                    levels[node_id] = levels[m]+1
    return levels
    



cpdef bfs(edge_index, node, limit, num_nodes):  # function for BFS
    cdef int** adj_list
    visited = [False for _ in range(num_nodes)]
    queue = Queue()
    levels = [limit for _ in range(num_nodes)]
    visited[node] = True
    levels[node] = 0
    queue.append(node)

    while not(queue.empty()):          # Creating loop to visit each node
        m = queue.pop(0)
        if levels[m] < limit:
            for node_id in graph[m]:
                if not(visited[node_id]):
                    visited[node_id] = True
                    queue.append(node_id)
                    levels[node_id] = levels[m]+1
    return levels

