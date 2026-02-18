# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:18:48 2025

@author: nguye
"""

#   test comment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from Consistent_Mass_matrix_2D9n_v2 import compute_q9_mass_matrix_numeric
# from Mass_matrix import compute_global_mass_matrix
from stiffness_matrix import global_stiffness_Q9geo_AGfield, compute_gauss_strain_stress_all_with_deformed
from ext_force_numba import global_external_force
from ext_force_distribution_force import global_external_distribution_force
import os


#i changed this
# =================================================================================================
# =                                     Generate Mesh                                             =
# =================================================================================================
def generate_q9_mesh(Lx, Ly, nx, ny):
    npx, npy = 2*nx + 1, 2*ny + 1
    coords = []
    for j in range(npy):
        y = Ly * j / (npy - 1)
        for i in range(npx):
            x = Lx * i / (npx - 1)
            coords.append([x, y])
    coords = np.array(coords)
    conn = []
    for ey in range(ny):
        for ex in range(nx):
            i0, j0 = ex * 2, ey * 2
            n = lambda dx, dy: (j0 + dy) * npx + (i0 + dx)
            elem_conn = [
                n(0, 0),   # node 1
                n(2, 0),   # node 2
                n(2, 2),   # node 3
                n(0, 2),   # node 4
                n(1, 0),   # node 5
                n(2, 1),   # node 6
                n(1, 2),   # node 7
                n(0, 1),   # node 8
                n(1, 1),   # node 9
            ]
            conn.append(elem_conn)
    connectivity = np.array(conn)
    return coords, connectivity

# (Optional) Add plotting function
def plot_nodes(coords, Lx, Ly):
    plt.figure(figsize=(3, 3))
    plt.scatter(coords[:,0], coords[:,1], s=80, color='dodgerblue', zorder=3)
    for idx, (x, y) in enumerate(coords):
        plt.text(x, y+0.03, f"{idx}", ha='center', va='bottom', fontsize=10, color='darkred')
    plt.title("Nodal Positions and Indices (2x2 Q9 mesh)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-0.1, Lx+0.1)
    plt.ylim(-0.1, Ly+0.1)
    plt.gca().set_aspect('equal')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

Lx, Ly = 10, 10
nx, ny = 40, 40
npx, npy = 2*nx+1, 2*ny+1  # nodes in x and y

coords, connectivity = generate_q9_mesh(Lx, Ly, nx, ny)
print(coords.shape)
# print(connectivity)

 ################## Plot element ##########################
# plot_nodes(coords, Lx, Ly)
offset_y = Ly * 0.005
plt.figure(figsize=(nx, ny))
plt.scatter(coords[:,0], coords[:,1], s=80, color='dodgerblue', zorder=3)
for idx, (x, y) in enumerate(coords):
    plt.text(x, y+offset_y, f"{idx}", ha='center', va='bottom', fontsize=10, color='darkred')

for elem, elem_conn in enumerate(connectivity):
    # Get the coordinates of the four corner nodes (assume order: 0-1-2-3 for Q9)
    corners = elem_conn[[0, 4, 1, 5, 2, 6, 3, 7, 0]]  # close the loop
    corner_coords = coords[corners]
    plt.plot(corner_coords[:,0], corner_coords[:,1], 'k-', lw=2, zorder=2)
    # Centroid (mean of all 4 corners)
    centroid = corner_coords[:-1].mean(axis=0)
    plt.text(centroid[0]+0.25*(Lx/nx), centroid[1]-0.25*(Ly/ny), f"e{elem}", ha='center', va='center',
             fontsize=12, fontweight='bold', color='darkblue', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.title("Mesh configuration", fontsize=24, fontweight='bold', fontname='Arial', pad=50)
plt.xlabel("x", fontsize=20, fontweight='medium', fontname='Arial')
plt.ylabel("y", fontsize=20, fontweight='medium', fontname='Arial')
plt.xlim(-0.1, Lx+0.1)
plt.ylim(-0.1, Ly+0.1)
plt.gca().set_aspect('equal')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()

left_edge_nodes = [j * npx for j in range(npy)]
right_edge_nodes = [j * npx + (npx-1) for j in range(npy)]
bottom_edge_nodes = [i for i in range(npx)]
top_edge_nodes = [(npy-1)*npx + i for i in range(npx)]

# print('Left edge:', left_edge_nodes)
# print('Right edge:', right_edge_nodes)
print('Bottom edge:', bottom_edge_nodes)
# print('Top edge:', top_edge_nodes)
     # 4------7------3
     # |             |
     # |             |
     # 8      9      6
     # |             |
     # |             |
     # 1------5------2


#%%
def get_top_edge_elements_between(coords, connectivity, nx, ny, x_min, x_max, tol=1e-12):
    """
    Return a list of (element_index, edge_id) for top-row elements whose top edge
    intersects [x_min, x_max]. Edge id = 4 for the top edge.
    """
    results = []
    ey = ny - 1  # top row
    for ex in range(nx):
        elem_idx = ey * nx + ex
        conn = connectivity[elem_idx]

        # Q9 ordering you used:
        # 0: (0,0)  1: (2,0)  2: (2,2)  3: (0,2)  4: (1,0)  5: (2,1)
        # 6: (1,2)  7: (0,1)  8: (1,1)
        # Top edge endpoints are node 4 (index 3) and node 3 (index 2):
        x_left  = coords[conn[3], 0]  # node 4 (top-left)
        x_right = coords[conn[2], 0]  # node 3 (top-right)

        # xmin_elem = min(x_left, x_right)
        # xmax_elem = max(x_left, x_right)

        # Inclusive overlap with tolerance
        xc = 0.5 * (x_left + x_right)
        if (xc >= x_min - tol) and (xc <= x_max + tol):
               results.append((elem_idx, 4))
    return results

# Example usage for the interval [4, 6]:
selected = get_top_edge_elements_between(coords, connectivity, nx, ny, 4.0, 6.0)
elem_ids_only = [e for e, edge in selected]

print("Top-edge elements intersecting [4,6]:", elem_ids_only)

#%%
# =================================================================================================
# =                          Define Element types, output directory                               =
# =================================================================================================    

 
# AG_list_type2 = [] ##....All quadratic
# AG_list_type1 = []
    
# AG_list_type2 = [94, 96] ##....10x10 mesh
# AG_list_type1 = [93, 95]

# AG_list_type2 = [388, 392] ##....20x20 mesh
# AG_list_type1 = [387, 391]

AG_list_type1 = [1575, 1583] ##....40x40 mesh
AG_list_type2 = [1576, 1584] ##....40x40 mesh

# AG_list_type1 = [6351, 6367] ##....80x80 mesh
# AG_list_type2 = [6352, 6368] ##....

nelem = nx * ny
etypes = [None] * nelem


m1, n1, m2, n2, ngauss = 1, 10, 1, 10, 10
for e in range(nelem):
    if e == 1575:
        etypes[e] = (m1, n1, m2, n2, ngauss, 1)
    elif e == 1576:
        etypes[e] = (m1, n1, m2, n2, ngauss, 2)
    elif e == 1583:
        etypes[e] = (m1, n1, m2, n2, ngauss, 1)
    elif e == 1584:
        etypes[e] = (m1, n1, m2, n2, ngauss, 2)
    else:
        etypes[e] = (1, 1, 1, 1, 3, 1) 

### For 80x80 mesh

# m1, n1, m2, n2, ngauss = 1, 5, 1, 5, 30 
# for e in range(nelem):
#     if e == 6351:
#         etypes[e] = (m1, n1, m2, n2, ngauss, 1)
#     elif e == 6352:
#         etypes[e] = (m1, n1, m2, n2, ngauss, 2)
#     elif e == 6367:
#         etypes[e] = (m1, n1, m2, n2, ngauss, 1)
#     elif e == 6368:
#         etypes[e] = (m1, n1, m2, n2, ngauss, 2)
#     else:
#         etypes[e] = (1, 1, 1, 1, 3, 1)        
        
# print(etypes)
def get_element_nodes(elem_id: int, connectivity: np.ndarray):
    if connectivity.ndim != 2 or connectivity.shape[1] != 9:
        raise ValueError("Expected connectivity to have shape (nelems, 9) for Q9 elements.")
    if not (0 <= elem_id < connectivity.shape[0]):
        raise IndexError(f"elem_id {elem_id} out of range [0, {connectivity.shape[0]-1}].")
    return connectivity[elem_id].copy()

#%% 
# =================================================================================================
# =                                      Stiffness Matrix                                         =
# =================================================================================================

ndof = 2
ncoords = 2
# Material properties
E = 1.0e9    # Young's modulus
nu = 0.3     # Poisson's ratio
rho1 = 1.0e3   # density
# Calculate Lamé parameters
mu = E / (2 * (1 + nu))

lumped_mass = True            # True: lumped, False: consistent 
planestrain = 1            # 1: plane strain, 0: plane stress
strain = None
materialprops = [mu, nu, planestrain]

K = global_stiffness_Q9geo_AGfield(coords, connectivity, etypes, ndof, materialprops)  

#%% 
# =================================================================================================
# =                                    Boundary Conditions                                        =
# =================================================================================================

# ----------------------------- Force Boundary Conditions -----------------------------
prescribe_by_node = {}
edge = 4 # (element index, edge number) (1,2,3,4):(left, bot, right, top)

def get_loaded_edges_top_between(coords, connectivity, nx, ny, x_min, x_max, edge, tol=1e-12):
    """
    Return [(elem_idx, edge)] for top-row elements whose x-span
    strictly overlaps (x_min, x_max). Elements that only touch
    at the boundary are excluded.
    """
    loaded_edges = []
    ey = ny - 1  # top row index
    for ex in range(nx):
        elem_idx = ey * nx + ex
        conn = connectivity[elem_idx]
        # take the 4 corner nodes
        corner_ids = [conn[0], conn[1], conn[2], conn[3]]
        x_coords = coords[corner_ids, 0]
        xmin_elem, xmax_elem = x_coords.min(), x_coords.max()

        # STRICT overlap (no equality) -> avoids nearby/touching elements
        if (xmax_elem > x_min + tol) and (xmin_elem < x_max - tol):
            loaded_edges.append((elem_idx, edge))

    return loaded_edges

def get_loaded_edges_all(coords, connectivity, nx, ny, edge):
    """
    Return [(elem_idx, edge)] for *all* elements in the mesh.
    """
    nelems = connectivity.shape[0]
    return [(elem_idx, edge) for elem_idx in range(nelems)]

def export_global_forces_to_excel(coords, f_global, filepath):
    """
    Export the full global force vector (all nodes) to Excel.
    Columns: node_id, x, y, Fx, Fy
    """
    n_nodes = f_global.shape[0]
    rows = []
    for i in range(n_nodes):
        rows.append({
            "node_id": i,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "Fx": float(f_global[i, 0]),
            "Fy": float(f_global[i, 1]),
        })
    df = pd.DataFrame(rows, columns=["node_id","x","y","Fx","Fy"])
    df.to_excel(filepath, index=False)
    return df

# ### Constant force distribution 
# traction = [0.0, -1000.0]  
# # (optional) sort by x to keep them ordered left→right
# top_nodes_loaded.sort(key=lambda n: coords[n, 0])
# print("Loaded top nodes:", top_nodes_loaded)

# # NOTE: Check prescribed force and displacement nodal set, avoid over constraints 
# loaded_edges = get_loaded_edges_top_between(coords, connectivity, nx, ny, 4.0, 6.0, edge)
# f = global_external_force(coords, connectivity, loaded_edges, traction, etypes, 2)
# f = f.reshape(-1)   # [Fx0, Fy0, Fx1, Fy1, ...]
# print("Loaded edge:", loaded_edges)
# print(f)

############ Distribution nodal force based on interpolating nodal values ############
# ----- endpoint-safe px(x) ----- 
def px(x, xmin=4.0, xmax=6.0, left_shift=0.003, right_shift=0.003):
    """
    p(x) = 1/sqrt(1-(x-5)^2) on [xmin, xmax] with endpoint truncation:
      - if x == xmin -> evaluate at xmin+left_shift (default 4.003)
      - if x == xmax -> evaluate at xmax-right_shift (default 5.997)
      - if x in (xmin, xmax) -> evaluate at x
      - else -> 0
    """
    if x < xmin or x > xmax:
        return 0.0
    # exact-endpoint handling (use isclose for safety)
    if np.isclose(x, xmin):
        x_eff = xmin + left_shift
    elif np.isclose(x, xmax):
        x_eff = xmax - right_shift
    else:
        x_eff = x
    denom = 1.0 - (x_eff - 5.0)**2
    # return -19.62410313*1.0e6 / np.sqrt(denom) if denom > 0.0 else 0.0
    return -19.62420313*1.0e6 / np.sqrt(denom) if denom > 0.0 else 0.0

def make_nodal_traction_top_px(coords, top_edge_nodes, component):
    """
    Returns traction_nodes: shape (Nnodes, 2).
    component: "x" -> put p(x) in tx; "y" -> put p(x) in ty.
    """
    Nnodes = coords.shape[0]
    traction_nodes = np.zeros((Nnodes, 2), dtype=np.float64)

    col = 0 if component == "x" else 1
    for nid in top_edge_nodes:
        x = coords[nid, 0]
        traction_nodes[nid, col] = px(x)
    return traction_nodes

def make_element_traction_arrays(connectivity, traction_nodes):
    """
    connectivity: (Ne, 9), traction_nodes: (Nnodes, 2)
    Returns elem_trac: (Ne, 9, 2) where elem_trac[e,i,:] = traction at local node i of element e
    """
    Ne = connectivity.shape[0]
    elem_trac = np.zeros((Ne, 9, 2), dtype=np.float64)
    for e in range(Ne):
        elem_trac[e, :, :] = traction_nodes[connectivity[e], :]
    return elem_trac
#%%
################################################################################
##############           Force Boundary conditions              ################
################################################################################


# foroutput = "force_control"
# # NOTE: Check prescribed force and displacement nodal set, avoid over constraints 
# loaded_edges = get_loaded_edges_top_between(coords, connectivity, nx, ny, 4.0, 6.0, edge)
# # loaded_edges = get_loaded_edges_all(coords, connectivity, nx, ny, edge)......Incorrect results
# # 1) Build nodal traction dataset (choose component)

# traction_nodes = make_nodal_traction_top_px(coords, top_edge_nodes, component="y") ##computed traction at nodes from built-in function 
 
# # print(traction_nodes)
# # 2) Per-element (9x2) arrays exactly like your example format


# elem_trac = make_element_traction_arrays(connectivity, traction_nodes)

# ## COMPUTE GLOBAL NODAL FORCE VECTOR
# elem_nodal_forces=[]
# f_global , elem_forces_agg  = global_external_distribution_force(coords, connectivity, loaded_edges, elem_trac, etypes)
# f = f_global.reshape(-1)   # [Fx0, Fy0, Fx1, Fy1, ...]
# print(traction_nodes)


# #####################            Save directory where this script lives             #############################
# script_dir = os.path.dirname(os.path.abspath(__file__))
# outdir = os.path.join(script_dir, f"{foroutput}_rigid_punch_({m1},{n1})_P=19.624e6_shift=0.003")
# os.makedirs(outdir, exist_ok=True)

# def export_elem_forces_all(elem_forces_agg, outpath):
#     """
#     elem_forces_agg: (Ne, 9, 2) accumulated per-element nodal forces (zeros for unloaded)
#     """
#     Ne = elem_forces_agg.shape[0]
#     rows = []
#     for e in range(Ne):
#         fe = elem_forces_agg[e]  # (9,2)
#         row = {"elem": e}
#         for i in range(9):
#             row[f"Fx{i+1}"] = float(fe[i, 0])
#             row[f"Fy{i+1}"] = float(fe[i, 1])
#         rows.append(row)
#     cols = ["elem"] + [c for i in range(9) for c in (f"Fx{i+1}", f"Fy{i+1}")]
#     df = pd.DataFrame(rows, columns=cols)
#     df.to_excel(outpath, index=False)
#     return df
# outfile = os.path.join(outdir, f"global_forces_{foroutput}_({m1},{n1}).xlsx")
# export_global_forces_to_excel(coords, f_global, outfile)
# print("Exported global forces to:", outfile)  


# # Export per-element nodal forces for ALL elements
# outfile_elem_all = os.path.join(outdir, f"element_nodal_forces_ALL_{foroutput}_({m1},{n1}).xlsx")
# export_elem_forces_all(elem_forces_agg, outfile_elem_all)
# print("Exported element nodal forces for all elements to:", outfile_elem_all)



####### Purely displacement control #######
foroutput = "disp_control"
f = np.zeros(K.shape[0])  # e.g., pure displacement-driven test

######################            Save directory where this script lives             #############################
script_dir = os.path.dirname(os.path.abspath(__file__))
# outdir = os.path.join(script_dir, f"{foroutput}_rigid_punch_({m1},{n1})_P=19.624e6_shift=0.003")
outdir = os.path.join(script_dir, f"2026_01_21_punch_results\{foroutput}({nx},{ny})_rigid_punch_({m1},{n1},{m2},{n2},{ngauss})_dz=-0.1")
os.makedirs(outdir, exist_ok=True)

#%% 
################################################################################
###########           Displacement Boundary Condition              #############
################################################################################
### Displacement BC

# pick top-edge nodes with 4 <= x <= 6
x_min, x_max = 4.0, 6.0
tol = 1e-12
top_nodes_loaded = [
    n for n in top_edge_nodes
    if (coords[n, 0] >= x_min - tol) and (coords[n, 0] <= x_max + tol)
]
#### Prescribed displacement
for n in top_nodes_loaded: # LOADED NODES
    prescribe_by_node.setdefault(n, {})
    # prescribe_by_node[n]['ux'] = 0.0
    prescribe_by_node[n]['uy'] = -0.1
print("boundary conds:" , prescribe_by_node)

###########################
# Bottom edge: fully fixed
for n in bottom_edge_nodes: # FIXED
    prescribe_by_node[n] = {'ux': 0.0, 'uy': 0.0}
    
#%% 
# =================================================================================================
# =                                            Solver                                             =
# =================================================================================================

# map node index -> (ux, uy) DOF indices
def node_dofs(n):
    return 2*n, 2*n+1

def build_dirichlet_from_nodes(prescribe_dict_by_node):
    """
    prescribe_dict_by_node: {node_id: {'ux': value_or_None, 'uy': value_or_None}}
      - put None for a dof you don't want to prescribe on that node
    Returns: dict {global_dof_index: value}
    """
    dbc = {}
    for n, spec in prescribe_dict_by_node.items():
        ux, uy = node_dofs(n)
        if spec.get('ux') is not None:
            dbc[ux] = float(spec['ux'])
        if spec.get('uy') is not None:
            dbc[uy] = float(spec['uy'])
    return dbc

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse.linalg import spsolve

def apply_dirichlet_elimination(K, f, prescribed_dofs):
    """
    K : (N,N) global stiffness (any; will be converted to CSR sparse)
    f : (N,)   RHS
    prescribed_dofs : dict {global_dof_index: value}
    returns: u (N,), reactions (N,)
    """
    # ---- make sparse CSR and proper dtypes ----
    if not isspmatrix(K):
        K = csr_matrix(K)
    else:
        K = K.tocsr()
    f = np.asarray(f, dtype=float)

    N = K.shape[0]
    u = np.zeros(N, dtype=float)

    # ---- no Dirichlet? solve whole system sparsely ----
    if not prescribed_dofs:
        u = spsolve(K, f)  # sparse direct solver
        reactions = K @ u - f
        return u, reactions

    # ---- indices/values for prescribed dofs ----
    # (dedupe with dict semantics; last wins)
    kv = {int(i): float(v) for i, v in prescribed_dofs.items()}
    p_idx = np.array(sorted(kv.keys()), dtype=int)
    up    = np.array([kv[i] for i in p_idx], dtype=float)
    u[p_idx] = up

    # ---- free set ----
    all_idx = np.arange(N, dtype=int)
    free = np.setdiff1d(all_idx, p_idx, assume_unique=False)

    # ---- reduced rhs: f_free -= K_free,pc * u_pc ----
    f_mod = f.copy()
    if p_idx.size:
        f_mod[free] -= K[free][:, p_idx] @ up

    # ---- reduced solve (sparse) ----
    Kff = K[free][:, free].tocsr()
    uf = spsolve(Kff, f_mod[free])

    # ---- assemble u and reactions ----
    u[free] = uf
    reactions = (K @ u) - f  # full-length residual; sums to net reactions
    return u, reactions

# def apply_dirichlet_elimination(K, f, prescribed_dofs):
#     N = K.shape[0]
#     u = np.zeros(N)

#     if len(prescribed_dofs) == 0:
#         # no Dirichlet — solve whole system (should have at least one support)
#         u[:] = np.linalg.solve(K, f)
#         reactions = K @ u - f
#         return u, reactions

#     # index sets
#     p_idx = np.array(sorted(prescribed_dofs.keys()), dtype=int)
#     u[p_idx] = np.array([prescribed_dofs[i] for i in p_idx], dtype=float)

#     all_idx = np.arange(N, dtype=int)
#     mask = np.ones(N, dtype=bool); mask[p_idx] = False
#     f_idx = all_idx[mask]

#     # partitions
#     Kff = K[np.ix_(f_idx, f_idx)]
#     Kfp = K[np.ix_(f_idx, p_idx)]
#     ff  = f[f_idx]
#     up  = u[p_idx]

#     # reduced solve
#     rhs = ff - Kfp @ up
#     uf = np.linalg.solve(Kff, rhs)

#     # assemble full u and reactions
#     u[f_idx] = uf
#     reactions = K @ u - f
#     return u, reactions                     
                      

###### To Solve ######
dbc = build_dirichlet_from_nodes(prescribe_by_node)
u, reactions = apply_dirichlet_elimination(K, f, dbc)

# reshape to (Nnodes, 2) if you like:
U_nodes = u.reshape(-1, 2)
# summed reactions (useful to check equilibrium)
Rx = reactions[0::2].sum()
Ry = reactions[1::2].sum()

print("ux range:", U_nodes[:,0].min(), U_nodes[:,0].max())
print("uy range:", U_nodes[:,1].min(), U_nodes[:,1].max())
print("Total reactions: Rx =", Rx, "Ry =", Ry)

#%%
#%% 
# =================================================================================================
# =                                       Post Processing                                         =
# =================================================================================================

def plot_deformed_points(coords, U_nodes, factor=1.0):
    """
    coords   : (N,2) array of original node coordinates
    U_nodes  : (N,2) array of nodal displacements
    factor   : scale factor for displacements
    """
    # original and deformed coordinates
    deformed = coords + factor * U_nodes

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')

    # plot original nodes (faded gray)
    plt.scatter(coords[:,0], coords[:,1], c='gray', s=10, alpha=0.5, label='original')

    # plot deformed nodes (blue)
    plt.scatter(deformed[:,0], deformed[:,1], c='blue', s=15, label=f'deformed (scale={factor})')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Nodal positions (original vs deformed)")
    plt.legend()
    plt.show()

# Example call after solving:
plot_deformed_points(coords, U_nodes, factor=10.0)

#%% Export nodal displacement

# --- reshape and split fields ---
Nnodes = coords.shape[0]
U_nodes = u.reshape(Nnodes, 2)
ux = U_nodes[:, 0]
uy = U_nodes[:, 1]

Rx = reactions[0::2]  # reaction force in x at each node
Ry = reactions[1::2]  # reaction force in y at each node

# optional magnitude
Rmag = np.hypot(Rx, Ry)

# --- build DataFrame and export ---
df = pd.DataFrame({
    "Node": np.arange(Nnodes, dtype=int),
    "x": coords[:, 0],
    "y": coords[:, 1],
    "ux": ux,
    "uy": uy,
    "Rx": Rx,
    "Ry": Ry,
    "Rmag": Rmag,
})

outfile = os.path.join(outdir, f"nodal_displacement_{foroutput}_({m1},{n1}).xlsx")
df.to_excel(outfile, index=False)
print("Saved:", outfile)
     

#%% Export stresses and strains at gauss points
gp = compute_gauss_strain_stress_all_with_deformed(coords, connectivity, etypes, u, materialprops)

#### Example DataFrame 
df = pd.DataFrame({
    "elem_id": gp["elem_id"],
    "xi": gp["xi_eta"][:,0], "eta": gp["xi_eta"][:,1],
    "x0": gp["xy0"][:,0], "y0": gp["xy0"][:,1],
    "x": gp["xy_def"][:,0], "y": gp["xy_def"][:,1],
    "detJ": gp["detJ"],
    "exx": gp["strain"][:,0], "eyy": gp["strain"][:,1], "gxy": gp["strain"][:,2],
    "sxx": gp["stress"][:,0], "syy": gp["stress"][:,1], "txy": gp["stress"][:,2],
    "szz": gp["sigma_zz"],
    "von_mises": gp["von_mises"],
})

outfile = os.path.join(outdir, f"gauss_strain_stress_deformed_{foroutput}_({m1},{n1}).xlsx")
df.to_excel(outfile, index=False)

#%%
def export_connectivity_to_excel(connectivity, coords=None, filepath="connectivity_Q9.xlsx"):
    """
    Export Q9 connectivity to Excel.

    connectivity: (Ne, 9) int array, each row is element -> 9 global node IDs
    coords: (Nnodes, 2) optional. If provided, also exports node coordinates in a 2nd sheet.
    filepath: output .xlsx path
    """
    Ne = connectivity.shape[0]
    df_conn = pd.DataFrame(
        connectivity,
        columns=[f"n{i}" for i in range(1, 10)]
    )
    df_conn.insert(0, "elem_id", np.arange(Ne, dtype=int))

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_conn.to_excel(writer, sheet_name="connectivity", index=False)

        if coords is not None:
            df_nodes = pd.DataFrame({
                "node_id": np.arange(coords.shape[0], dtype=int),
                "x": coords[:, 0],
                "y": coords[:, 1],
            })
            df_nodes.to_excel(writer, sheet_name="nodes", index=False)

    return df_conn

# Example usage:
outfile_conn = os.path.join(outdir, f"connectivity_Q9_{nx}x{ny}.xlsx")
export_connectivity_to_excel(connectivity, coords=coords, filepath=outfile_conn)
print("Saved connectivity to:", outfile_conn)
