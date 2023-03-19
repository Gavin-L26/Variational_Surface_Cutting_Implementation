import igl
import meshplot as mp
import numpy as np
import scipy as sp
import math 
from src.dual_length import get_dual_length
from src.separate_mesh import get_separate_mesh
from src.dirichlet_energy import get_dirichlet_energy
from src.length_energy import get_length_energy

# Inputs:
#   v  #v by 3 list of mesh vertices
#   f  #f by 3 list of mesh faces (must be triangles)
#   d  #v list of signed distance value from the cut curve (this value = 0 when the corresponding vertex is part of the cut curve)
#   iter number of iteration to run the program
#   bnd  #bnd list of cut curve vertex indices into v
# Outputs:
#   new_d  #v list of signed distance value from the new optimized cut curve (this value = 0 when the corresponding vertex is part of the cut curve)

def do_cut_step(v, f, d, iter, bnd):
    v_iter = v
    f_iter = f
    d_iter = d

    p2 = mp.plot(v_iter, f_iter, d_iter, shading={"wireframe": False})
    p2.add_points(v_iter[bnd], shading={"point_size": 0.2})

    for i in range(iter):
        (v_iter, 
        f_iter, 
        d_iter, 
        v_l, 
        f_l, 
        d_l, 
        v_r, 
        f_r, 
        d_r, 
        n_l, 
        n_r, 
        n_b) = get_separate_mesh(v_iter, f_iter, d_iter)


        dual_length_l = get_dual_length(f_l, v_l, n_b, n_l)
        dual_length_r = get_dual_length(f_r, v_r, n_b, n_r)

        dir_grad_l = get_dirichlet_energy(f_l, v_l, n_b, n_l, dual_length_l)
        dir_grad_r = get_dirichlet_energy(f_r, v_r, n_b, n_r, dual_length_r)

        dldn_l = get_length_energy(v_l, f_l, d_l, n_l)
        dldn_r = get_length_energy(v_r, f_r, d_r, n_r)

        bnd_grad_l = dir_grad_l + dldn_l
        bnd_grad_r = dir_grad_r + dldn_r

        bnd_grad = bnd_grad_l - bnd_grad_r

        bnd_idx = np.arange(n_l+n_r, v_iter.shape[0])
        uv = igl.harmonic_weights(v_iter, f_iter, bnd_idx, bnd_grad, 1)

        tao = 0.0001
        d_iter -= tao * uv

        L = igl.cotmatrix(v, f)

        iden = np.identity(L.shape[0])

        lhs = iden + tao * L
        d_iter = np.linalg.solve(lhs, d_iter)
        
        adj_list = igl.adjacency_list(f_iter)
        possible_bnd = [] 
        d_iter_bool = d_iter > 0
        for i in range(v_iter.shape[0]):
            if d_iter_bool[i]:
                for j in range(len(adj_list[i])):
                    if not d_iter_bool[adj_list[i][j]]:
                        if any(not d_iter_bool[k] for k in adj_list[adj_list[i][j]]):
                            possible_bnd.append(i)
        
        possible_bnd = np.array(possible_bnd)

        vt = np.arange(v_iter.shape[0])
        new_d = igl.exact_geodesic(v_iter, f_iter, possible_bnd, vt)

        for i in range(d_iter_bool.shape[0]):
            if not d_iter_bool[i]:
                new_d[i] = -new_d[i]
        
        d_iter = new_d

    p2 = mp.plot(v_iter, f_iter, d_iter, shading={"wireframe": False})
    p2.add_points(v_iter[possible_bnd], shading={"point_size": 0.2})
    return d_iter