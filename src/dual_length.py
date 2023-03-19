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
# Outputs:
#   dual_length  n_bnd list of dual length of a boundary vertex

def get_dual_length(sub_f, sub_v, n_bnd, n_internal):
    bnd_e = igl.boundary_facets(sub_f)
    E_bnd = 0
    dual_length = np.zeros(n_bnd)

    for edge in bnd_e:
        edge_val = np.array([sub_v[edge[0]], sub_v[edge[1]]])
        energy = np.linalg.norm(edge_val)
        E_bnd += energy
        dual_length[edge[0]-n_internal] += energy
        dual_length[edge[1]-n_internal] += energy
        
     #Maybe divde by total surface area?
    # print(dual_length)
    for i in range(dual_length.shape[0]):
        if dual_length[i] == 0:
            dual_length[i] = 1
    
    # p = mp.plot(sub_v, sub_f, shading={"wireframe": False})
    # p.add_edges(sub_v, bnd_e, shading={"line_color": "red"})
    return dual_length