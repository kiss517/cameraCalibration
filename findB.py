import numpy as np

def findM_B(V):
    assert len(V) >= 6, "At least 3 pictures are required to compute the homography."

    V_matrix = np.array(V).reshape(-1,6)
    U, S, VT = np.linalg.svd(V_matrix)
    b = VT.T[:, -1]
    B = [[b[0], b[1], b[2]], 
         [b[1], b[3], b[4]], 
         [b[2], b[4], b[5]]]
    B = np.array(B,dtype=np.float64)
    # if B[2, 2] != 0:
    #     B = B / B[2, 2]
    
    return B
# V = [
#     [1, 2, 3, 4, 5, 6],
#     [2, 3, 4, 5, 6, 7],
#     [3, 4, 5, 6, 7, 8],
#     [4, 5, 6, 7, 8, 9],
#     [5, 6, 7, 8, 9, 10],
#     [6, 7, 8, 9, 10, 11]
# ]

# B = findM_B(V)
# print(B)
# B=[B[0,0],B[0,1],B[0,2],B[1,1],B[1,2],B[2,2]]
# print(np.dot(V,B))