from queue import Queue
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython



cdef link ** linked_list(int length):
    cdef int i
    cdef link ** link_list = <link **> PyMem_Malloc(length * sizeof(link*))
    for i in range(length):
        link_list[i] = NULL
    return link_list

def bfs(graph, node, limit, num_nodes):  # function for BFS

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
