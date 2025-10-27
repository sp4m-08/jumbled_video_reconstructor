import numpy as np

def tsp_reconstruction(similarity_matrix):
    N = similarity_matrix.shape[0]

    # Step 1: Nearest Neighbor heuristic
    unvisited = set(range(N))
    current = 0
    order = [current]
    unvisited.remove(current)

    while unvisited:
        next_frame = max(unvisited, key=lambda x: similarity_matrix[current, x])
        order.append(next_frame)
        unvisited.remove(next_frame)
        current = next_frame

    # Step 2: 2-opt refinement
    improved = True
    while improved:
        improved = False
        for i in range(1, N-2):
            for j in range(i+1, N):
                if j+1 >= N:
                    continue
                a, b = order[i-1], order[i]
                c, d = order[j], order[j+1]
                current_score = similarity_matrix[a, b] + similarity_matrix[c, d]
                swapped_score = similarity_matrix[a, c] + similarity_matrix[b, d]
                if swapped_score > current_score:
                    order[i:j+1] = reversed(order[i:j+1])
                    improved = True
    return order
