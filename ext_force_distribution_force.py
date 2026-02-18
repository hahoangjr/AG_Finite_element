# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 10:24:09 2025

@author: nguye
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from stiffness_matrix import dN_dxi_AGc3, dN_deta_AGc3, dN_dxi_AGc4, dN_deta_AGc4

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

#%%
# ---- force kernel: analytic Jacobian via AG derivatives; nodal traction (9x2) ----
@njit(fastmath=True, cache=True)
def compute_q9_ag_force_numba(coords, edge, traction_9x2, eletype, pts, wts):
    """
    coords        : (9,2) element nodal coordinates
    edge          : 1(left), 2(bottom), 3(right), 4(top)
    traction_9x2  : (9,2) nodal traction [tx_j, ty_j] at each element node j
    eletype       : (m1, n1, m2, n2, ngauss, AGtype)
    pts, wts      : Gauss points/weights on [-1,1]
    returns       : fext (9,2) consistent nodal force for this edge
    """
    nNodes = 9
    fext = np.zeros((nNodes, 2))
    m1, n1, m2, n2, ngauss, AGtype = eletype

    for i in range(nNodes):
        fx_i = 0.0
        fy_i = 0.0

        for k in range(ngauss):
            s = pts[k]; w = wts[k]

            # edge parametrization (s=eta on 1,3; s=xi on 2,4)
            if edge == 1:      xi, eta = -1.0, s
            elif edge == 2:    xi, eta = s,  -1.0
            elif edge == 3:    xi, eta =  1.0, s
            else:              xi, eta = s,   1.0  # edge==4

            # field shapes at (xi,eta)
            if AGtype == 2:
                Nvals  = shape_functions_AGc4(xi, eta, m1, n1, m2, n2)
                dNdxi  = dN_dxi_AGc4(xi, eta, m1, n1, m2, n2)   # uses your imported AGc4 d/dxi
                dNdeta = dN_deta_AGc4(xi, eta, m1, n1, m2, n2)  # uses your imported AGc4 d/deta
            else:
                Nvals  = shape_functions_AGc3(xi, eta, m1, n1, m2, n2)
                dNdxi  = dN_dxi_AGc3(xi, eta, m1, n1, m2, n2)
                dNdeta = dN_deta_AGc3(xi, eta, m1, n1, m2, n2)

            # interpolate traction at Gauss point: t_gp = Σ N_j * t_j
            tx_gp = 0.0; ty_gp = 0.0
            for j in range(nNodes):
                tx_gp += Nvals[j] * traction_9x2[j, 0]
                ty_gp += Nvals[j] * traction_9x2[j, 1]

            # analytic tangent: dx/ds = Σ (dN/dξ or dN/dη) * x_j
            dxds = 0.0; dyds = 0.0
            if edge == 2 or edge == 4:   # s = xi
                for j in range(nNodes):
                    dxds += dNdxi[j]  * coords[j, 0]
                    dyds += dNdxi[j]  * coords[j, 1]
            else:                        # s = eta
                for j in range(nNodes):
                    dxds += dNdeta[j] * coords[j, 0]
                    dyds += dNdeta[j] * coords[j, 1]
            jac = (dxds*dxds + dyds*dyds)**0.5  # |dx/ds|

            # consistent nodal force contribution for node i
            scale = Nvals[i] * jac * w
            fx_i += scale * tx_gp
            fy_i += scale * ty_gp

        fext[i, 0] = fx_i
        fext[i, 1] = fy_i

    return fext

def compute_q9_ag_force(coords, edge, traction, eletype):
    
    # # ensure traction is a float64 NumPy array of shape (2,)
    # traction_arr = np.asarray(traction, dtype=np.float64)
    # if traction_arr.shape != (2,):
    #     raise ValueError("traction must be length-2 (tx, ty).")

    m1, n1, m2, n2, ngauss, AGtype = eletype
    pts, wts = leggauss(ngauss)
    return compute_q9_ag_force_numba(coords, edge, traction, eletype, pts, wts)

def global_external_distribution_force(
    coords,
    connectivity, loaded_edges, traction, etype,
    ndof=2
):
    n_nodes = coords.shape[0]
    f_global = np.zeros((n_nodes, ndof))
    nelems  = connectivity.shape[0]
    
    elem_forces_agg = np.zeros((nelems, 9, ndof), dtype=np.float64)
    
    for elem_idx, edge_num in loaded_edges:
        elem_conn = connectivity[elem_idx]
        elem_coords = coords[elem_conn, :]
        ngauss  = etype[elem_idx][4]
        xi_pts, xi_wts   = leggauss(ngauss)
        eta_pts, eta_wts = leggauss(ngauss)
        

        # Compute element external force for this edge
        f_elem = compute_q9_ag_force(elem_coords, edge_num, traction[elem_idx], etype[elem_idx])  # (9,2)
        # print("elem_id = ", elem_idx)
        # print("f_elem = ",f_elem)
        # append (no np.stack)
        # elem_nodal_forces.append(f_elem.copy())
        elem_forces_agg[elem_idx] += f_elem
        
        # Scatter assembly
        for local_idx, global_idx in enumerate(elem_conn):
            f_global[global_idx] += f_elem[local_idx]
    return f_global, elem_forces_agg

# coord = np.array([[ 5.75 ,  9.75 ],[ 6.   ,  9.75 ], [ 6.   , 10.   ],
#        [ 5.75 , 10.   ],[ 5.875,  9.75 ], [ 6.   ,  9.875], 
#        [ 5.875, 10.   ],[ 5.75 ,  9.875], [ 5.875,  9.875]], dtype=np.float64)

# edge = 4

# # traction = np.array([[px1, py1],[px2, py2],[px3, py3],
# #                      [px4, py4],[px5, py5],[px6, py6],
# #                      [px7, py7],[px8, py8],[px9, py9]  ], dtype=np.float64)

# traction = np.array([[0, 0],[0, 0],[0, 12.9196],
#                      [0, 1.51186],[0, 0],[0, 0],
#                      [0, 2.06559],[0, 0],[0, 0]  ], dtype=np.float64)
# etype = (1, 5, 1, 5, 30, 1)

# m1, n1, m2, n2, ngauss, AGtype = etype
# pts, wts = leggauss(ngauss)

# nodalf = compute_q9_ag_force_numba(coord, 4, traction, etype, pts, wts)

# print(nodalf)

#####Test codes

#%%
# def generate_q9_mesh(Lx, Ly, nx, ny):
#     npx, npy = 2*nx + 1, 2*ny + 1
#     coords = []
#     for j in range(npy):
#         y = Ly * j / (npy - 1)
#         for i in range(npx):
#             x = Lx * i / (npx - 1)
#             coords.append([x, y])
#     coords = np.array(coords)
#     conn = []
#     for ey in range(ny):
#         for ex in range(nx):
#             i0, j0 = ex * 2, ey * 2
#             n = lambda dx, dy: (j0 + dy) * npx + (i0 + dx)
#             elem_conn = [
#                 n(0, 0),   # node 1
#                 n(2, 0),   # node 2
#                 n(2, 2),   # node 3
#                 n(0, 2),   # node 4
#                 n(1, 0),   # node 5
#                 n(2, 1),   # node 6
#                 n(1, 2),   # node 7
#                 n(0, 1),   # node 8
#                 n(1, 1),   # node 9
#             ]
#             conn.append(elem_conn)
#     connectivity = np.array(conn)
#     return coords, connectivity

# # (Optional) Add plotting function
# def plot_nodes(coords, Lx, Ly):
#     plt.figure(figsize=(3, 3))
#     plt.scatter(coords[:,0], coords[:,1], s=80, color='dodgerblue', zorder=3)
#     for idx, (x, y) in enumerate(coords):
#         plt.text(x, y+0.03, f"{idx}", ha='center', va='bottom', fontsize=10, color='darkred')
#     plt.title("Nodal Positions and Indices (2x2 Q9 mesh)")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.xlim(-0.1, Lx+0.1)
#     plt.ylim(-0.1, Ly+0.1)
#     plt.gca().set_aspect('equal')
#     plt.grid(True, which='both', linestyle='--', alpha=0.5)
#     plt.show()

# Lx, Ly = 10, 10
# nx, ny = 40, 40
# npx, npy = 2*nx+1, 2*ny+1  # nodes in x and y

# coords, connectivity = generate_q9_mesh(Lx, Ly, nx, ny)
# print(coords.shape)
# print(connectivity)

#   ################## Plot element ##########################
# # plot_nodes(coords, Lx, Ly)
# offset_y = Ly * 0.005
# plt.figure(figsize=(nx, ny))
# plt.scatter(coords[:,0], coords[:,1], s=80, color='dodgerblue', zorder=3)
# for idx, (x, y) in enumerate(coords):
#     plt.text(x, y+offset_y, f"{idx}", ha='center', va='bottom', fontsize=10, color='darkred')

# for elem, elem_conn in enumerate(connectivity):
#     # Get the coordinates of the four corner nodes (assume order: 0-1-2-3 for Q9)
#     corners = elem_conn[[0, 4, 1, 5, 2, 6, 3, 7, 0]]  # close the loop
#     corner_coords = coords[corners]
#     plt.plot(corner_coords[:,0], corner_coords[:,1], 'k-', lw=2, zorder=2)
#     # Centroid (mean of all 4 corners)
#     centroid = corner_coords[:-1].mean(axis=0)
#     plt.text(centroid[0]+0.25*(Lx/nx), centroid[1]-0.25*(Ly/ny), f"e{elem}", ha='center', va='center',
#               fontsize=12, fontweight='bold', color='darkblue', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# plt.title("Mesh configuration", fontsize=24, fontweight='bold', fontname='Arial', pad=50)
# plt.xlabel("x", fontsize=20, fontweight='medium', fontname='Arial')
# plt.ylabel("y", fontsize=20, fontweight='medium', fontname='Arial')
# plt.xlim(-0.1, Lx+0.1)
# plt.ylim(-0.1, Ly+0.1)
# plt.gca().set_aspect('equal')
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.show()

# left_edge_nodes = [j * npx for j in range(npy)]
# right_edge_nodes = [j * npx + (npx-1) for j in range(npy)]
# bottom_edge_nodes = [i for i in range(npx)]
# top_edge_nodes = [(npy-1)*npx + i for i in range(npx)]

# # print('Left edge:', left_edge_nodes)
# # print('Right edge:', right_edge_nodes)
# # print('Bottom edge:', bottom_edge_nodes)
# # print('Top edge:', top_edge_nodes)
#       # 4------7------3
#       # |             |
#       # |             |
#       # 8      9      6
#       # |             |
#       # |             |
#       # 1------5------2

# # ----- endpoint-safe px(x) -----
# def px_of_x_truncated(x, xmin=4.0, xmax=6.0, left_shift=0.003, right_shift=0.003):
#     """
#     p(x) = 1/sqrt(1-(x-5)^2) on [xmin, xmax] with endpoint truncation:
#       - if x == xmin -> evaluate at xmin+left_shift (default 4.003)
#       - if x == xmax -> evaluate at xmax-right_shift (default 5.997)
#       - if x in (xmin, xmax) -> evaluate at x
#       - else -> 0
#     """
#     if x < xmin or x > xmax:
#         return 0.0

#     # exact-endpoint handling (use isclose for safety)
#     if np.isclose(x, xmin):
#         x_eff = xmin + left_shift
#     elif np.isclose(x, xmax):
#         x_eff = xmax - right_shift
#     else:
#         x_eff = x

#     denom = 1.0 - (x_eff - 5.0)**2
#     return 1.0 / np.sqrt(denom) if denom > 0.0 else 0.0

# def make_nodal_traction_top_px(coords, top_edge_nodes, component):
#     """
#     Returns traction_nodes: shape (Nnodes, 2).
#     component: "x" -> put p(x) in tx; "y" -> put p(x) in ty.
#     """
#     Nnodes = coords.shape[0]
#     traction_nodes = np.zeros((Nnodes, 2), dtype=np.float64)

#     col = 0 if component == "x" else 1
#     for nid in top_edge_nodes:
#         x = coords[nid, 0]
#         traction_nodes[nid, col] = px_of_x_truncated(x)
#     return traction_nodes

# # ----------------------------- 2) Per-element (9x2) arrays -----------------------------
# def make_element_traction_arrays(connectivity, traction_nodes):
#     """
#     connectivity: (Ne, 9), traction_nodes: (Nnodes, 2)
#     Returns elem_trac: (Ne, 9, 2) where elem_trac[e,i,:] = traction at local node i of element e
#     """
#     Ne = connectivity.shape[0]
#     elem_trac = np.zeros((Ne, 9, 2), dtype=np.float64)
#     for e in range(Ne):
#         elem_trac[e, :, :] = traction_nodes[connectivity[e], :]
#     return elem_trac

# def get_loaded_edges_top_between(coords, connectivity, nx, ny, x_min, x_max, edge):
#     """
#     Return [(elem_idx, edge_num)] for top-row elements whose x-span overlaps (x_min, x_max).
#     Edge numbering: 1=left, 2=bottom, 3=right, 4=top
#     """
#     loaded_edges = []
#     ey = ny - 1                # top row index
#     for ex in range(nx):
#         elem_idx = ey * nx + ex
#         # Four corner nodes of this element
#         conn = connectivity[elem_idx]
#         corner_ids = [conn[0], conn[1], conn[2], conn[3]]
#         x_coords = coords[corner_ids, 0]
#         xmin_elem, xmax_elem = x_coords.min(), x_coords.max()
#         if (xmax_elem > x_min) and (xmin_elem < x_max):  # overlap test
#             loaded_edges.append((elem_idx, edge))           # top edge
#     return loaded_edges

# nelem = nx * ny
# etypes = [None] * nelem


# for e in range(nelem):
#     if e == 1575:
#         etypes[e] = (0.5, 2, 0.5, 2, 30, 1)
#     elif e == 1576:
#         etypes[e] = (1, 1, 0.5, 2, 30, 2)
#     elif e == 1583:
#         etypes[e] = (1, 1, 0.5, 2, 30, 1)
#     elif e == 1584:
#         etypes[e] = (0.5, 2, 0.5, 2, 30, 2)
#     else:
#         etypes[e] = (1, 1, 1, 1, 3, 1)
# edge = 4
# # NOTE: Check prescribed force and displacement nodal set, avoid over constraints 
# loaded_edges = get_loaded_edges_top_between(coords, connectivity, nx, ny, 4.0, 6.0, edge)

# # 1) Build nodal traction dataset (choose component)
# traction_nodes = make_nodal_traction_top_px(coords, top_edge_nodes, component="y")
# # 2) Per-element (9x2) arrays exactly like your example format
# elem_trac = make_element_traction_arrays(connectivity, traction_nodes)

# f_global = global_external_distribution_force(coords, connectivity, loaded_edges, elem_trac, etypes)

# #%%
# # node_ids = np.arange(6561, dtype=int).reshape(-1, 1)
# # f_global_withid = np.hstack((node_ids, f_global)) 

# # print(f_global_withid)
# #%%









