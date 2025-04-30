from typing import List

import numpy as np
import time
from openai import OpenAI

class TopologyLabel:
    def __init__(self, nid: int, collider=False, chain=False, fork=False):
        self.nid : int = nid
        self.collider : bool = collider
        self.chain : bool = chain
        self.fork : bool = fork

    def __str__(self):
        return (f"{self.nid}: {{collider = {self.collider}, chain = {self.chain}, "
                f"fork = {self.fork}}}")

    def __repr__(self):
        return str(self)

class GraphNode:
    def __init__(self):
        self.predecessors : List[int] = []
        self.out_degree = 0

    def __str__(self):
        return f"{{incoming = {self.predecessors}, n_outgoing = {self.out_degree}}}"

    def __repr__(self):
        return str(self)

def labeler(adj: np.ndarray) -> List[TopologyLabel]:
    # asserts the layout of the input adjacency matrix
    assert len(adj.shape) == 2 and adj.shape[0] == adj.shape[1]
    n = adj.shape[0]
    result : List[TopologyLabel] = []
    al : List[GraphNode] = []
    enqueued: List[bool] = []

    # Initialize data structures
    for i in range(n):
        al.append(GraphNode())
        enqueued.append(False)
        result.append(TopologyLabel(i))

    # Build adjacency list
    vertex = 0
    for row in adj:
        descendant = 0
        for connection in row:
            if descendant != vertex and connection == 1:
                al[descendant].predecessors.append(vertex)
                al[vertex].out_degree += 1
            descendant += 1
        vertex += 1

    q: List[int] = []
    for i in range(n):
        if al[i].out_degree == 0:
            q.append(i)
            enqueued[i] = True

    current = 0
    while current < n:
        node = q[current]
        if len(al[node].predecessors) > 1:
            result[node].collider = True
        elif len(al[node].predecessors) == 1 and al[node].out_degree >= 1:
            result[node].chain = True
        if al[node].out_degree > 1:
            result[node].fork = True
        current += 1
        for p in al[node].predecessors:
            if not enqueued[p]:
                q.append(p)
                enqueued[p] = True
    return result

if __name__ == '__main__':
    client = OpenAI()

    start_time = time.perf_counter()
    response = client.responses.create(
        model="gpt-4.1",
        input="Write a one-sentence bedtime story about a unicorn."
    )
    x = response.output_text
    end_time = time.perf_counter()
    duration = (end_time - start_time)
    print("Completes in {}s".format(duration))
    print(response.output_text)
    # 0 -> 1
    # 0 -> 2
    # matrix = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    # mal = labeler(matrix)
    # print(mal)
