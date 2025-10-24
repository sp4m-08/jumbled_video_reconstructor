import numpy as np
def greedy_reconstruction(similarity_matrix):
    n = len(similarity_matrix)
    visited = np.zeros(n,dtype=bool)
    sequence = [0]
    visited[0] = True
    #greedy algorithm based on similarity scores between frames (travelling salesman)
    for _ in range(1,n):
        last = sequence[-1]
        scores = similarity_matrix[last].copy()
        scores[visited] = -1
        next_idx = np.argmax(scores)
        sequence.append(next_idx)
        visited[next_idx] = True
        
    return sequence

def tsp_reconstruction(similarity_matrix):
    """
    Reconstruct frame order using TSP approximation (nearest neighbor + 2-opt).
    similarity_matrix: NxN numpy array where entry (i,j) = SSIM between frame i and frame j
    Returns:
        order: list of frame indices in reconstructed order
    """
    N = similarity_matrix.shape[0]

    # Step 1: Nearest Neighbor heuristic
    unvisited = set(range(N))
    current = 0  # start at first frame
    order = [current]
    unvisited.remove(current)

    while unvisited:
        # Pick the next frame with max similarity
        next_frame = max(unvisited, key=lambda x: similarity_matrix[current, x])
        order.append(next_frame)
        unvisited.remove(next_frame)
        current = next_frame

    # Step 2: 2-opt local refinement
    improved = True
    while improved:
        improved = False
        for i in range(1, N-2):
            for j in range(i+1, N):
                # Current edges: i-1 -> i, j -> j+1
                # Swap edges if it improves total similarity
                if j+1 >= N:
                    continue
                a, b = order[i-1], order[i]
                c, d = order[j], order[j+1]
                current_score = similarity_matrix[a,b] + similarity_matrix[c,d]
                swapped_score = similarity_matrix[a,c] + similarity_matrix[b,d]
                if swapped_score > current_score:
                    # Perform swap
                    order[i:j+1] = reversed(order[i:j+1])
                    improved = True
    return order

        