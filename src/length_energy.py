import igl
import meshplot as mp
import numpy as np
import scipy as sp
import math 

# Inputs:
#   sub_v  #sub_v by 3 list of submesh vertices
#   sub_f  #sub_f by 3 list of submesh faces (must be triangles)
#   sub_d  #sub_v list of submesh signed distance value
#   n_internal number of internal vertices
# Outputs:
#   bnd_l_grad gradient of edge length energy at the boundary

def get_length_energy(sub_v, sub_f, sub_d, n_internal):
    # #cotmatrix
    # L = igl.cotmatrix(v, f)
    # L = -L

    # #massmatrix
    # M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    
    # iden = np.identity(L.shape[0])
    # lhs = iden + tao * a_L * L
    # rhs = M * d

    # new_d = np.linalg.solve(lhs, rhs)
    area = igl.doublearea(sub_v, sub_f) / 2
    dldn_sum = np.zeros(sub_v.shape[0])
    dldn_weight = np.zeros(sub_v.shape[0])

    for face in sub_f:
        for i in range(3):
            target = (i+1)%3

            edge_length = np.linalg.norm(sub_v[face[i]] - sub_v[face[target]])
            
            dldn_sum[face[i]] -= sub_d[face[i]]
            dldn_weight[face[i]] += edge_length

            dldn_sum[face[target]] -= sub_d[face[target]]
            dldn_weight[face[target]] += edge_length
    
    for i in range(dldn_weight.shape[0]):
        if dldn_weight[i] == 0:
            dldn_weight[i] = 1
    
    weight = 0.1
    dldn = weight * dldn_sum / dldn_weight
    bnd_l_grad = dldn[n_internal:]

    return bnd_l_grad