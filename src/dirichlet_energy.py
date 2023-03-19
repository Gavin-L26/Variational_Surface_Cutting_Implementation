import igl
import meshplot as mp
import numpy as np
import scipy as sp
import math 

# Inputs:
#   sub_v  #sub_v by 3 list of submesh vertices
#   sub_f  #sub_f by 3 list of submesh faces (must be triangles)
#   n_bnd  number of bound vertices
#   n_internal number of internal vertices
#   dual_length n_bnd list of dual length of a boundary vertex
# Outputs:
#   bnd_grad gradient of dirichlet_energy at the boundary

def get_dirichlet_energy(sub_f, sub_v, n_bnd, n_internal, dual_length):

    #Face Mass Matrix
    area = igl.doublearea(sub_v, sub_f) / 2
    W_f = sp.sparse.lil_matrix((sub_v.shape[0], sub_f.shape[0]))

    for i in range(sub_f.shape[0]):
        for j in range(3):
            W_f[sub_f[i][j], i] = area[i] / 3
    
    #Curvature
    internal_angles = igl.internal_angles(sub_v, sub_f)
    ver_angle_sum = np.zeros(sub_v.shape[0])
    tri_angle_sum = np.ones(sub_f.shape[0])
    tri_angle_sum *= math.pi

    for i in range(sub_f.shape[0]):
        for j in range(3):
            ver_angle_sum[sub_f[i][j]] += internal_angles[i][j] 

    for i in range(sub_f.shape[0]):
        for j in range(3):
            tri_angle_sum[i] -=  2 * math.pi * internal_angles[i][j] / ver_angle_sum[sub_f[i][j]]

    K = tri_angle_sum / area

    #Construct RHS
    rhs = W_f * K
    rhs_interior = rhs[:n_internal]

    #Cot Matrix
    L = igl.cotmatrix(sub_v, sub_f)
    #change to positive semidefinite
    L = -L

    L_ii = L[0:n_internal, 0:n_internal]

    #solve for interior value first
    u_i = sp.sparse.linalg.spsolve(L_ii, rhs_interior)

    distortion = np.zeros(sub_v.shape[0])
    for i in range(n_internal):
        distortion[i] = u_i[i]

    #boundary vertices normals
    for i in range(sub_f.shape[0]):
        for j in range(3):
            other_a = (j+1)%3
            other_b = (j+2)%3

            if (sub_f[i][j] < n_internal):
                continue
            
            if(sub_f[i][other_a] < n_internal):
                v1 = sub_v[sub_f[i][j]] - sub_v[sub_f[i][other_b]]
                v2 = sub_v[sub_f[i][other_a]] - sub_v[sub_f[i][other_b]]
                cotan = (np.dot(v1, v2)) / (np.linalg.norm(np.cross(v1, v2)))
                distortion[sub_f[i][j]] += 0.5 * cotan * distortion[sub_f[i][other_a]]
            if(sub_f[i][other_b] < n_internal):
                v1 = sub_v[sub_f[i][other_b]] - sub_v[sub_f[i][other_a]]
                v2 = sub_v[sub_f[i][j]] - sub_v[sub_f[i][other_a]]
                cotan = (np.dot(v1, v2))/ (np.linalg.norm(np.cross(v1, v2)))
                distortion[sub_f[i][j]] += 0.5 * cotan * distortion[sub_f[i][other_b]]
            
            distortion[sub_f[i][j]] += (area[i] / 3) * K[i]
    
    for i in range(sub_v.shape[0]):
        if(i >= n_internal): 
            distortion[i] /= dual_length[i - n_internal];

    weight = 3

    bnd_grad = weight * distortion[n_internal:] * distortion[n_internal:]

    return bnd_grad