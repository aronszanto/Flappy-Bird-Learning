import util


class Fringe:

    def __init__(self, s):
        self.structure = s()
        assert ('push' in dir(s) and 'pop' in dir(s) and 'isEmpty' in dir(s))

    def push(self, item, base, cost):
        self.structure.push(item) if cost == 0 else self.structure.push(item, base + cost)

    def pop(self):
        return self.structure.pop()

    def isEmpty(self):
        return self.structure.isEmpty()


def search(structure, cost_function=None):
    import node_util
    if cost_function is None:
        cost_function = lambda successor: 0

    fringe = Fringe(structure)

    # holds visited nodes
    visited = {}
    # (k,v) pair where k = node n1, v = parent of n1
    parent = {}
    start = node_util.getStart(), []
    visited[start[0]] = 0

    for successor in node_util.getSuccessors(start[0]):
        fringe.push((successor[0], [successor[1]]), 0, cost_function(successor))
        visited[successor[0]] = successor[2]  # update best cost to successors
    called = 0
    while not fringe.isEmpty():
        called += 1
        cur = fringe.pop()
        if called % 10000 == 0:
            print called, cur
        if node_util.isGoalState(cur[0]):
            return cur[1]
        else:
            for successor in node_util.getSuccessors(cur[0]):
                if successor[0] in visited and visited[successor[0]] > (visited[cur[0]] + successor[2]):
                    curpath=cur[1]
                    newpath=curpath + [successor[1]]
                    fringe.push((successor[0], newpath), visited[
                                cur[0]], cost_function(successor))
                    visited[successor[0]]=visited[cur[0]] + successor[2]
                elif successor[0] not in visited:
                    curpath=cur[1]
                    newpath=curpath + [successor[1]]
                    visited[successor[0]]=visited[cur[0]] + successor[2]
                    fringe.push((successor[0], newpath), visited[
                                cur[0]], cost_function(successor))
from node_util import IMAGES
def heuristic(state):
    state = state[0]
    playerMidPos = state.x + IMAGES['player'][0].get_width() / 2
    for upipe, lpipe in zip(state.upipes, state.lpipes):
        pipeMidPos = upipe['x'] + IMAGES['pipe'][0].get_width() / 2
        if pipeMidPos > playerMidPos:
            y_avg = float(upipe['y'] + lpipe['y']) / 2
            return abs(state.y - y_avg) + abs(state.x - pipeMidPos)

print search(util.PriorityQueue, heuristic)
