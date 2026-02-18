# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 15:32:52 2025

@author: nguye
"""

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss

# ================== SHAPE FUNCTIONS ==================
@njit(fastmath=True, cache=True)
def gamma(eta, m, n):
    return np.abs(1 - (1 - ((eta + 1) / 2)**2)**m)**n

@njit(fastmath=True, cache=True)
def NAG3_one(eta, m, n):
    G0 = gamma(0, m, n)
    G_eta = gamma(eta, m, n)
    return (-G0 + (G0 - 1)*eta + G_eta) / (1 - 2 * G0)

@njit(fastmath=True, cache=True)
def NAG3_two(eta, m, n):
    G0 = gamma(0, m, n)
    G_eta = gamma(eta, m, n)
    return (1 + eta - 2 * G_eta) / (1 - 2 * G0)

@njit(fastmath=True, cache=True)
def NAG3_three(eta, m, n):
    G0 = gamma(0, m, n)
    G_eta = gamma(eta, m, n)
    return (-G0 - G0 * eta + G_eta) / (1 - 2 * G0)

@njit(fastmath=True, cache=True)
def shape_functions_AGc3(xi, eta, m1, n1, m2, n2):
    N = np.zeros(9)
    N[0] = NAG3_one(xi, m1, n1)   * NAG3_one(eta, m2, n2)
    N[1] = NAG3_three(xi, m1, n1) * NAG3_one(eta, m2, n2)
    N[2] = NAG3_three(xi, m1, n1) * NAG3_three(eta, m2, n2)
    N[3] = NAG3_one(xi, m1, n1)   * NAG3_three(eta, m2, n2)
    N[4] = NAG3_two(xi, m1, n1)   * NAG3_one(eta, m2, n2)
    N[5] = NAG3_three(xi, m1, n1) * NAG3_two(eta, m2, n2)
    N[6] = NAG3_two(xi, m1, n1)   * NAG3_three(eta, m2, n2)
    N[7] = NAG3_one(xi, m1, n1)   * NAG3_two(eta, m2, n2)
    N[8] = NAG3_two(xi, m1, n1)   * NAG3_two(eta, m2, n2)
    return N

@njit(fastmath=True, cache=True)
def shape_functions_AGc4(xi, eta, m1, n1, m2, n2):
    N = np.zeros(9)
    N[0] = NAG3_three(-xi, m1, n1)   * NAG3_one(eta, m2, n2)
    N[1] = NAG3_one(-xi, m1, n1) * NAG3_one(eta, m2, n2)
    N[2] = NAG3_one(-xi, m1, n1) * NAG3_three(eta, m2, n2)
    N[3] = NAG3_three(-xi, m1, n1)   * NAG3_three(eta, m2, n2)
    N[4] = NAG3_two(-xi, m1, n1)   * NAG3_one(eta, m2, n2)
    N[5] = NAG3_one(-xi, m1, n1) * NAG3_two(eta, m2, n2)
    N[6] = NAG3_two(-xi, m1, n1)   * NAG3_three(eta, m2, n2)
    N[7] = NAG3_three(-xi, m1, n1)   * NAG3_two(eta, m2, n2)
    N[8] = NAG3_two(-xi, m1, n1)   * NAG3_two(eta, m2, n2)
    return N
# ================== Q9 GEOMETRY (bi-quadratic Lagrange) ==================
@njit(fastmath=True, cache=True)
def q9_1d_L(ξ):
    # Nodes at ξ = -1, 0, +1
    L1 = 0.5 * ξ * (ξ - 1.0)   # N at -1
    L2 = 1.0 - ξ*ξ             # N at  0
    L3 = 0.5 * ξ * (ξ + 1.0)   # N at +1
    return L1, L2, L3

@njit(fastmath=True, cache=True)
def q9_1d_dL(ξ):
    dL1 = ξ - 0.5
    dL2 = -2.0 * ξ
    dL3 = ξ + 0.5
    return dL1, dL2, dL3

@njit(fastmath=True, cache=True)
def q9_geom_shapes_derivs(xi, eta):
    """
    Q9 geometry shapes N^g_a, and derivatives wrt (ξ,η).
    Ordering (a=0..8):
      0:(-1,-1)=L1(ξ)L1(η)
      1:(+1,-1)=L3(ξ)L1(η)
      2:(+1,+1)=L3(ξ)L3(η)
      3:(-1,+1)=L1(ξ)L3(η)
      4:( 0,-1)=L2(ξ)L1(η)
      5:(+1, 0)=L3(ξ)L2(η)
      6:( 0,+1)=L2(ξ)L3(η)
      7:(-1, 0)=L1(ξ)L2(η)
      8:( 0, 0)=L2(ξ)L2(η)
    """
    Lx1, Lx2, Lx3 = q9_1d_L(xi)
    Ly1, Ly2, Ly3 = q9_1d_L(eta)
    dLx1, dLx2, dLx3 = q9_1d_dL(xi)
    dLy1, dLy2, dLy3 = q9_1d_dL(eta)

    N = np.empty(9)
    dN_dxi  = np.empty(9)
    dN_deta = np.empty(9)

    # corners
    N[0] = Lx1*Ly1; dN_dxi[0] = dLx1*Ly1; dN_deta[0] = Lx1*dLy1
    N[1] = Lx3*Ly1; dN_dxi[1] = dLx3*Ly1; dN_deta[1] = Lx3*dLy1
    N[2] = Lx3*Ly3; dN_dxi[2] = dLx3*Ly3; dN_deta[2] = Lx3*dLy3
    N[3] = Lx1*Ly3; dN_dxi[3] = dLx1*Ly3; dN_deta[3] = Lx1*dLy3
    # midsides
    N[4] = Lx2*Ly1; dN_dxi[4] = dLx2*Ly1; dN_deta[4] = Lx2*dLy1
    N[5] = Lx3*Ly2; dN_dxi[5] = dLx3*Ly2; dN_deta[5] = Lx3*dLy2
    N[6] = Lx2*Ly3; dN_dxi[6] = dLx2*Ly3; dN_deta[6] = Lx2*dLy3
    N[7] = Lx1*Ly2; dN_dxi[7] = dLx1*Ly2; dN_deta[7] = Lx1*dLy2
    # center
    N[8] = Lx2*Ly2; dN_dxi[8] = dLx2*Ly2; dN_deta[8] = Lx2*dLy2

    return N, dN_dxi, dN_deta


# ----------- Numba-enabled Q9 AG force (pass quadrature from Python) -----------

# @njit(fastmath=True, cache=True)
# def compute_q9_ag_force_numba(coords, edge, traction, eletype, pts, wts):
#     nNodes = 9
#     fext = np.zeros((nNodes, 2))
#     m1, n1, m2, n2, ngauss, AGtype = eletype
#     for i in range(nNodes):
#         fe = np.zeros(2)
#         for k in range(ngauss):
#             s = pts[k]
#             w = wts[k]
#             if edge == 1:  # left
#                 xi = -1.0
#                 eta = s
#             elif edge == 2:  # bottom
#                 xi = s
#                 eta = -1.0
#             elif edge == 3:  # right
#                 xi = 1.0
#                 eta = s
#             elif edge == 4:  # top
#                 xi = s
#                 eta = 1.0
#             else:
#                 continue  # (cannot raise inside njit!)

#             if AGtype == 2:      
#                 # --- Field derivatives wrt parent: AG/Q9 you defined ---
#                 Nvals = shape_functions_AGc4(xi, eta, m1, n1, m2, n2)
#             else:
#                 # --- Field derivatives wrt parent: AG/Q9 you defined ---
#                 Nvals = shape_functions_AGc3(xi, eta, m1, n1, m2, n2)
                
#             h = 1e-8
#             if edge in (1, 3):
#                 xi1, eta1 = xi, s + h
#                 xi2, eta2 = xi, s - h
#             else:
#                 xi1, eta1 = s + h, eta
#                 xi2, eta2 = s - h, eta

#             x1 = 0.0
#             y1 = 0.0
#             x2 = 0.0
#             y2 = 0.0
#             # Nj1 = shape_functions(xi1, eta1, m1, n1, m2, n2)
#             # Nj2 = shape_functions(xi2, eta2, m1, n1, m2, n2)
#             if AGtype == 2:      
#                 # --- Field derivatives wrt parent: AG/Q9 you defined ---
#                 Nj1 = shape_functions_AGc4(xi1, eta1, m1, n1, m2, n2)
#                 Nj2 = shape_functions_AGc4(xi2, eta2, m1, n1, m2, n2)
#             else:
#                 # --- Field derivatives wrt parent: AG/Q9 you defined ---
#                 Nj1 = shape_functions_AGc3(xi1, eta1, m1, n1, m2, n2)
#                 Nj2 = shape_functions_AGc3(xi2, eta2, m1, n1, m2, n2)
                
                
#             for j in range(nNodes):
#                 x1 += Nj1[j] * coords[j, 0]
#                 y1 += Nj1[j] * coords[j, 1]
#                 x2 += Nj2[j] * coords[j, 0]
#                 y2 += Nj2[j] * coords[j, 1]
#             dx = (x1 - x2) / (2 * h)
#             dy = (y1 - y2) / (2 * h)
#             jac = np.sqrt(dx ** 2 + dy ** 2)
#             Ni = Nvals[i]
#             fe += Ni * jac * w * traction
#         fext[i, :] = fe
#     return fext

@njit(fastmath=True, cache=True)
def compute_q9_ag_force_numba(coords, edge, traction2, eletype, pts, wts):
    """
    traction2: np.ndarray shape (2,), float64
    """
    nNodes = 9
    fext = np.zeros((nNodes, 2))
    m1, n1, m2, n2, ngauss, AGtype = eletype

    for i in range(nNodes):
        fe = np.zeros(2)
        for k in range(ngauss):
            s = pts[k]
            w = wts[k]
            if edge == 1:      xi, eta = -1.0, s
            elif edge == 2:    xi, eta = s, -1.0
            elif edge == 3:    xi, eta =  1.0, s
            elif edge == 4:    xi, eta = s,  1.0
            else:
                continue

            if AGtype == 2:
                Nvals = shape_functions_AGc4(xi, eta, m1, n1, m2, n2)
            else:
                Nvals = shape_functions_AGc3(xi, eta, m1, n1, m2, n2)

            # --- your finite-diff jacobian (kept as-is) ---
            h = 1e-8
            if edge in (1, 3):
                xi1, eta1 = xi, s + h
                xi2, eta2 = xi, s - h
            else:
                xi1, eta1 = s + h, eta
                xi2, eta2 = s - h, eta

            x1 = 0.0; y1 = 0.0
            x2 = 0.0; y2 = 0.0
            if AGtype == 2:
                Nj1 = shape_functions_AGc4(xi1, eta1, m1, n1, m2, n2)
                Nj2 = shape_functions_AGc4(xi2, eta2, m1, n1, m2, n2)
            else:
                Nj1 = shape_functions_AGc3(xi1, eta1, m1, n1, m2, n2)
                Nj2 = shape_functions_AGc3(xi2, eta2, m1, n1, m2, n2)

            for j in range(nNodes):
                x1 += Nj1[j] * coords[j, 0]
                y1 += Nj1[j] * coords[j, 1]
                x2 += Nj2[j] * coords[j, 0]
                y2 += Nj2[j] * coords[j, 1]

            dx = (x1 - x2) / (2 * h)
            dy = (y1 - y2) / (2 * h)
            jac = np.sqrt(dx*dx + dy*dy)

            Ni = Nvals[i]
            scale = Ni * jac * w

            # traction2 is np.ndarray (2,), so this is vector math
            fe[0] += scale * traction2[0]
            fe[1] += scale * traction2[1]

        fext[i, :] = fe
    return fext

# ----------- Python wrapper for easy use (precompute gauss pts/weights) -----------

# def compute_q9_ag_force(coords, edge, traction, eletype):
#     m1, n1, m2, n2, ngauss, AGtype = eletype
#     from numpy.polynomial.legendre import leggauss
#     pts, wts = leggauss(ngauss)
#     return compute_q9_ag_force_numba(coords, edge, traction, eletype, pts, wts)

def compute_q9_ag_force(coords, edge, traction, eletype):
    # ensure traction is a float64 NumPy array of shape (2,)
    traction_arr = np.asarray(traction, dtype=np.float64)
    if traction_arr.shape != (2,):
        raise ValueError("traction must be length-2 (tx, ty).")

    m1, n1, m2, n2, ngauss, AGtype = eletype
    pts, wts = leggauss(ngauss)
    return compute_q9_ag_force_numba(coords, edge, traction_arr, eletype, pts, wts)

def global_external_force(
    coords,
    connectivity, loaded_edges, traction, etype,
    ndof=2
):
    n_nodes = coords.shape[0]
    f_global = np.zeros((n_nodes, ndof))
    for elem_idx, edge_num in loaded_edges:
        elem_conn = connectivity[elem_idx]
        elem_coords = coords[elem_conn, :]
        ngauss  = etype[elem_idx][4]
        xi_pts, xi_wts   = leggauss(ngauss)
        eta_pts, eta_wts = leggauss(ngauss)
        

        # Compute element external force for this edge
        f_elem = compute_q9_ag_force(elem_coords, edge_num, traction, etype[elem_idx])  # (9,2)
        # Scatter assembly
        for local_idx, global_idx in enumerate(elem_conn):
            f_global[global_idx] += f_elem[local_idx]
    return f_global
# # ---------------------- USAGE EXAMPLE ----------------------
# import time
# start = time.perf_counter()
# if __name__ == "__main__":
#     coords = np.array([
#         [-1, -1], [1, -1], [1, 1],
#         [-1, 1], [0, -1], [1, 0],
#         [0, 1], [-1, 0], [0, 0]
#     ], dtype=np.float64)
#     edge = 3
#     traction = np.array([1.0, 0.0])
#     eletype = [1, 1, 1, 1, 5]  # m1, n1, m2, n2, ngauss

#     fvec = compute_q9_ag_force(coords, edge, traction, eletype)
#     print(fvec)
    
# end = time.perf_counter()
# print(f"Lambda (Python) time: {end-start:.4f} s")
    

# ------------------- Example usage -------------------
# import time
# start = time.perf_counter()
# if __name__ == "__main__":
#     # 1 element, unit square, all edges load [1, 0]
#     coords = np.array([
#         [-1, -1], [1, -1], [1, 1],
#         [-1, 1], [0, -1], [1, 0],
#         [0, 1], [-1, 0], [0, 0]
#     ], dtype=np.float64)
#     connectivity = np.array([
#         [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     ])
#     # List all edges: [(element_idx, edge_num)]
#     loaded_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
#     traction = np.array([1.0, 0.0])
#     AG_list = [0]
#     AG_type = [1, 1, 1, 1, 5]
#     quadratic_type = [1, 1, 1, 1, 5]

#     f_global = global_external_force(
#         coords, connectivity, loaded_edges, traction,
#         AG_list, AG_type, quadratic_type, ndof=2
#     )
#     print(f_global)

# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.6f} seconds")
    
