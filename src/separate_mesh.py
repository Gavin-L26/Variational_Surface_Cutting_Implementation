import igl
import meshplot as mp
import numpy as np
import scipy as sp
import math 

# Inputs:
#   v  #v by 3 list of mesh vertices
#   f  #f by 3 list of mesh faces (must be triangles)
#   d  #v list of signed distance value from the cut curve (this value = 0 when the corresponding vertex is part of the cut curve)
# Outputs:
#   ordered_v  #v by 3 list of mesh vertices that is ordered by their signed distance value to: outer vertices -> inner vertices -> bounding vertices
#   ordered_f  #f by 3 list of mesh faces that is ordered in outer faces -> inner faces
#   ordered_d  #v list of signed distance value from the cut curve. It is ordered similarly to ordered_v
#   sub_v_l  nl+nb by 3 list of mesh outer vertices and boundary vertices
#   left_f  #left_f by 3 list of outer mesh faces
#   d_l  nl list of signed distance value of the outer mesh faces
#   sub_v_r  nr+nb by 3 list of mesh inner vertices and boundary vertices
#   right_f  #right_f by 3 list of inner mesh faces
#   d_r  nr list of signed distance value of the inner mesh faces
#   n_l  number of outer vertices
#   n_r  number of inner vertices
#   n_b  number of boundary vertices

def get_separate_mesh(v, f, d):

    #Reorder mesh

    #Left and right vertex used with index into global v
    left_ver_idx = []
    right_ver_idx = []
    bnd_ver_idx = []
    
    for face in f:
        if all(d[p] >= 0 for p in face):
            for i in range(3):
                if (face[i] not in left_ver_idx):
                    if (d[face[i]] == 0 and face[i] not in bnd_ver_idx):
                        bnd_ver_idx.append(face[i])
                    elif (d[face[i]] > 0):
                        left_ver_idx.append(face[i])

        else:
            for i in range(3):
                if (face[i] not in right_ver_idx):
                    if (d[face[i]] == 0 and face[i] not in bnd_ver_idx):
                        bnd_ver_idx.append(face[i])
                    elif (d[face[i]] < 0):
                        right_ver_idx.append(face[i])
    
    left_ver_idx = np.array(left_ver_idx).astype(int)
    right_ver_idx = np.array(right_ver_idx).astype(int)
    bnd_ver_idx = np.array(bnd_ver_idx).astype(int)

    n_l = left_ver_idx.shape[0]
    n_r = right_ver_idx.shape[0]
    n_b = bnd_ver_idx.shape[0]

    ver_index = np.concatenate([left_ver_idx, right_ver_idx, bnd_ver_idx], 0).astype(int)

    ordered_v = v[ver_index]
    ordered_d = d[ver_index]
    ordered_f = []
    
    #Separate Mesh
    left_f = []
    right_f = []

    ver_index_l = np.concatenate([left_ver_idx, bnd_ver_idx], 0)
    ver_index_r = np.concatenate([right_ver_idx, bnd_ver_idx], 0)

    sub_v_l = v[ver_index_l]
    sub_v_r = v[ver_index_r]
    d_l = d[ver_index_l]
    d_r = d[ver_index_r]

    for face in f:
        sub_face_idx = np.empty(3)
        for i in range(3):
            sub_face_idx[i] = np.where(ver_index == face[i])[0]
        ordered_f.append(sub_face_idx.astype(int))
        if all(d[p] >= 0 for p in face):
            for i in range(3):
                sub_face_idx[i] = np.where(ver_index_l == face[i])[0]
            left_f.append(sub_face_idx.astype(int))

        else:
            for i in range(3):
                sub_face_idx[i] = np.where(ver_index_r == face[i])[0]
            right_f.append(sub_face_idx.astype(int))
    
    ordered_f = np.array(ordered_f)
    left_f = np.array(left_f)
    right_f = np.array(right_f)
    
    # mp.plot(sub_v_l, left_f, d_l, shading={"wireframe": False})
    # mp.plot(sub_v_r, right_f, d_r, shading={"wireframe": False})

    return (ordered_v, ordered_f, ordered_d, sub_v_l, left_f, d_l, sub_v_r, right_f, d_r, n_l, n_r, n_b)