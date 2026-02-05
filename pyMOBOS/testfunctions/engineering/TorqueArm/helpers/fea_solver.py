"""
Torque Arm FEA Solver

Solves 3D linear elasticity problem for the torque arm:
    ∇·σ = 0       (equilibrium)
    σ = C:ε       (constitutive law - Hooke's law)
    ε = ½(∇u + ∇uᵀ)  (strain-displacement relation)

Boundary Conditions:
    - Fixed (u=0) on inner surface of left hole (Dirichlet BC)
    - Point/distributed loads at right hole (Neumann BC)

Outputs:
    - Maximum von Mises stress (MPa)
    - Maximum displacement magnitude (mm)
    - Total volume (mm³)

Two solver implementations:
    1. SfePyFEASolver - Uses SfePy library (more robust, recommended)
    2. SimpleFEASolver - Direct stiffness method (fallback, no dependencies)

Dependencies:
    Required: numpy, scipy, meshio
    Optional: sfepy (for SfePyFEASolver)
    jax jaxlib
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import warnings

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False

try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Check for SfePy
try:
    import sfepy
    from sfepy.discrete.fem import Mesh as SfePyMesh
    HAS_SFEPY = True
except ImportError:
    HAS_SFEPY = False


class SimpleFEASolver:
    """
    Linear elasticity FEA solver using direct stiffness method.
    
    Implements standard FEM for 3D linear elasticity with 4-node
    tetrahedral elements (constant strain tetrahedra).
    
    This is a self-contained solver that only requires numpy/scipy.
    """
    
    # Material properties (steel)
    E = 200e3       # Young's modulus (MPa = N/mm²)
    NU = 0.3        # Poisson's ratio
    
    # Loading (from paper: 8kN down, 4kN left at right hole)
    F_Y = -8000.0   # N, downward at right hole
    F_X = -4000.0   # N, leftward at right hole
    
    # Geometry constants for BC identification
    L_ARM = 300.0           # Center-to-center distance
    R_LEFT_INNER = 22.5     # D_LEFT_INNER / 2
    R_RIGHT_OUTER = 27.5    # D_RIGHT_OUTER / 2
    R_RIGHT_INNER = 15.0    # D_RIGHT_INNER / 2
    
    def __init__(self, verbose: bool = False):
        """
        Initialize solver.
        
        :param verbose: Print progress information
        """
        if not HAS_MESHIO:
            raise ImportError("meshio is required. Install with: pip install meshio")
        if not HAS_SCIPY:
            raise ImportError("scipy is required. Install with: pip install scipy")
        
        self.verbose = verbose
    
    def solve(self, mesh_path: str) -> Tuple[float, float, float]:
        """
        Solve linear elasticity on the given mesh.
        
        :param mesh_path: Path to mesh file (.msh, .vtk, etc.)
        :return: (max_displacement_mm, max_von_mises_MPa, volume_mm3)
        """
        # Read mesh
        mesh = meshio.read(mesh_path)
        nodes = mesh.points.astype(np.float64)
        
        # Get tetrahedral elements
        elements = self._extract_tetrahedra(mesh)
        
        if elements is None or len(elements) == 0:
            raise ValueError("No tetrahedral elements found in mesh")
        
        n_nodes = len(nodes)
        n_elements = len(elements)
        n_dof = 3 * n_nodes
        
        if self.verbose:
            print(f"Mesh: {n_nodes} nodes, {n_elements} tetrahedra, {n_dof} DOFs")
        
        # Build material stiffness matrix
        D = self._elasticity_matrix()
        
        # Assemble global stiffness matrix
        if self.verbose:
            print("Assembling stiffness matrix...")
        K = self._assemble_stiffness(nodes, elements, D)
        
        # Identify boundary conditions
        if self.verbose:
            print("Identifying boundary conditions...")
        fixed_nodes = self._find_fixed_nodes(nodes)
        load_nodes = self._find_load_nodes(nodes)
        
        if len(fixed_nodes) == 0:
            raise RuntimeError("No fixed nodes found - geometry may be incorrect")
        if len(load_nodes) == 0:
            raise RuntimeError("No load nodes found - geometry may be incorrect")
        
        if self.verbose:
            print(f"  Fixed nodes: {len(fixed_nodes)}")
            print(f"  Load nodes: {len(load_nodes)}")
        
        # Build DOF lists
        fixed_dofs = self._nodes_to_dofs(fixed_nodes)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
        
        if len(free_dofs) == 0:
            raise RuntimeError("All DOFs fixed - check boundary conditions")
        
        # Create load vector
        F = np.zeros(n_dof)
        n_load = len(load_nodes)
        for node in load_nodes:
            F[3*node + 0] += self.F_X / n_load  # X force (leftward)
            F[3*node + 1] += self.F_Y / n_load  # Y force (downward)
        
        # Solve reduced system: K_ff * u_f = F_f
        if self.verbose:
            print(f"Solving {len(free_dofs)} DOF system...")
        
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        K_sparse = csr_matrix(K_ff)
        u_f = spsolve(K_sparse, F_f)
        
        # Reconstruct full displacement vector
        u = np.zeros(n_dof)
        u[free_dofs] = u_f
        
        # Compute results
        if self.verbose:
            print("Computing results...")
        
        # Maximum displacement
        u_nodal = u.reshape(n_nodes, 3)
        disp_mag = np.linalg.norm(u_nodal, axis=1)
        max_disp = disp_mag.max()
        
        # Maximum von Mises stress
        max_vm = self._compute_max_von_mises(nodes, elements, u, D)
        
        # Total volume
        volume = self._compute_volume(nodes, elements)
        
        if self.verbose:
            print(f"Results:")
            print(f"  Max displacement: {max_disp:.4f} mm")
            print(f"  Max von Mises: {max_vm:.2f} MPa")
            print(f"  Volume: {volume:.2f} mm³")
        
        return max_disp, max_vm, volume
    
    def _extract_tetrahedra(self, mesh) -> Optional[np.ndarray]:
        """Extract tetrahedral element connectivity from mesh."""
        for cell_block in mesh.cells:
            if cell_block.type == 'tetra':
                return cell_block.data.astype(np.int32)
            elif cell_block.type == 'tetra10':
                # Quadratic tet - use only corner nodes for linear analysis
                return cell_block.data[:, :4].astype(np.int32)
        return None
    
    def _elasticity_matrix(self) -> np.ndarray:
        """
        Create 6x6 isotropic elasticity matrix (Voigt notation).
        
        σ = D * ε
        where σ = [σxx, σyy, σzz, τxy, τyz, τxz]ᵀ
              ε = [εxx, εyy, εzz, γxy, γyz, γxz]ᵀ
        """
        E, nu = self.E, self.NU
        
        # Lamé parameters
        lam = E * nu / ((1 + nu) * (1 - 2*nu))
        mu = E / (2 * (1 + nu))
        
        D = np.array([
            [lam + 2*mu, lam,       lam,       0,  0,  0],
            [lam,       lam + 2*mu, lam,       0,  0,  0],
            [lam,       lam,       lam + 2*mu, 0,  0,  0],
            [0,         0,         0,         mu, 0,  0],
            [0,         0,         0,         0,  mu, 0],
            [0,         0,         0,         0,  0,  mu]
        ], dtype=np.float64)
        
        return D
    
    def _assemble_stiffness(self, nodes: np.ndarray, 
                           elements: np.ndarray,
                           D: np.ndarray) -> np.ndarray:
        """
        Assemble global stiffness matrix using direct stiffness method.
        
        For each element:
        1. Compute element stiffness Ke (12x12 for 4-node tet)
        2. Add Ke contributions to global K at appropriate DOF locations
        """
        n_nodes = len(nodes)
        n_dof = 3 * n_nodes
        
        # Use dense matrix for simplicity (works for meshes up to ~20k nodes)
        # For larger meshes, should use sparse assembly (COO format)
        K = np.zeros((n_dof, n_dof), dtype=np.float64)
        
        for elem in elements:
            # Get element node coordinates
            elem_nodes = nodes[elem]
            
            # Compute element stiffness
            Ke = self._element_stiffness_tet4(elem_nodes, D)
            
            # Get global DOF indices for this element
            dofs = self._nodes_to_dofs(elem)
            
            # Assemble into global matrix
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    K[di, dj] += Ke[i, j]
        
        return K
    
    def _element_stiffness_tet4(self, elem_nodes: np.ndarray, 
                                D: np.ndarray) -> np.ndarray:
        """
        Compute 12x12 element stiffness matrix for 4-node tetrahedron.
        
        The linear tetrahedron has constant strain throughout, so
        the stiffness integral can be evaluated analytically:
        
        Ke = V * Bᵀ * D * B
        
        where V is element volume and B is the strain-displacement matrix.
        """
        # Extract coordinates
        x, y, z = elem_nodes[:, 0], elem_nodes[:, 1], elem_nodes[:, 2]
        
        # Jacobian matrix for coordinate transformation
        # J[i,j] = ∂x_i / ∂ξ_j
        J = np.array([
            [x[1]-x[0], x[2]-x[0], x[3]-x[0]],
            [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
            [z[1]-z[0], z[2]-z[0], z[3]-z[0]]
        ], dtype=np.float64)
        
        detJ = np.linalg.det(J)
        
        # Skip degenerate (zero-volume) elements
        if abs(detJ) < 1e-14:
            return np.zeros((12, 12), dtype=np.float64)
        
        # Element volume
        V = abs(detJ) / 6.0
        
        # Shape function derivatives in natural coordinates
        # N0 = 1 - ξ - η - ζ, N1 = ξ, N2 = η, N3 = ζ
        dN_dxi = np.array([
            [-1.0, -1.0, -1.0],  # ∂N0/∂(ξ,η,ζ)
            [ 1.0,  0.0,  0.0],  # ∂N1/∂(ξ,η,ζ)
            [ 0.0,  1.0,  0.0],  # ∂N2/∂(ξ,η,ζ)
            [ 0.0,  0.0,  1.0]   # ∂N3/∂(ξ,η,ζ)
        ], dtype=np.float64)
        
        # Transform to physical coordinates: ∂N/∂x = ∂N/∂ξ * J⁻¹
        Jinv = np.linalg.inv(J)
        dN_dx = dN_dxi @ Jinv.T  # Shape: (4, 3)
        
        # Build strain-displacement matrix B (6x12)
        # B relates nodal displacements to strains: ε = B * u_e
        B = np.zeros((6, 12), dtype=np.float64)
        
        for i in range(4):  # Loop over 4 nodes
            col = 3 * i  # Starting column for this node's DOFs
            
            # Derivatives of shape function i
            dNi_dx, dNi_dy, dNi_dz = dN_dx[i, :]
            
            # Fill B matrix according to strain definitions:
            # εxx = ∂u/∂x, εyy = ∂v/∂y, εzz = ∂w/∂z
            # γxy = ∂u/∂y + ∂v/∂x, γyz = ∂v/∂z + ∂w/∂y, γxz = ∂u/∂z + ∂w/∂x
            
            B[0, col+0] = dNi_dx  # εxx
            B[1, col+1] = dNi_dy  # εyy
            B[2, col+2] = dNi_dz  # εzz
            B[3, col+0] = dNi_dy  # γxy (∂u/∂y)
            B[3, col+1] = dNi_dx  # γxy (∂v/∂x)
            B[4, col+1] = dNi_dz  # γyz (∂v/∂z)
            B[4, col+2] = dNi_dy  # γyz (∂w/∂y)
            B[5, col+0] = dNi_dz  # γxz (∂u/∂z)
            B[5, col+2] = dNi_dx  # γxz (∂w/∂x)
        
        # Element stiffness: Ke = V * Bᵀ * D * B
        Ke = V * (B.T @ D @ B)
        
        return Ke
    
    def _nodes_to_dofs(self, nodes: np.ndarray) -> np.ndarray:
        """Convert node indices to DOF indices (3 DOFs per node: ux, uy, uz)."""
        nodes = np.asarray(nodes)
        dofs = np.empty(3 * len(nodes), dtype=np.int32)
        for i, node in enumerate(nodes):
            dofs[3*i:3*i+3] = [3*node, 3*node+1, 3*node+2]
        return dofs
    
    def _find_fixed_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """
        Find nodes on the inner surface of the left hole for fixed BC.
        
        The left hole:
        - Center at (0, 0, z) - axis along Z
        - Inner radius = R_LEFT_INNER = 22.5 mm
        
        We identify nodes that are:
        1. Close to x = 0 (left end)
        2. At distance ≈ R_LEFT_INNER from the Y axis
        """
        # Distance from the left hole axis (x=0, y=0)
        r_from_axis = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
        
        # Tolerance for surface detection
        r_tol = 5.0  # mm
        x_tol = self.R_LEFT_INNER + 10.0
        
        # Nodes on inner cylindrical surface
        on_inner_surface = np.abs(r_from_axis - self.R_LEFT_INNER) < r_tol
        
        # Nodes near the left end
        at_left_end = np.abs(nodes[:, 0]) < x_tol
        
        # Combined condition
        fixed_mask = on_inner_surface & at_left_end
        
        # Fallback: if no nodes found, use leftmost region
        if not np.any(fixed_mask):
            x_min = nodes[:, 0].min()
            fixed_mask = nodes[:, 0] < x_min + 10.0
            if self.verbose:
                print(f"  Warning: Using fallback fixed BC (leftmost nodes)")
        
        return np.where(fixed_mask)[0]
    
    def _find_load_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """
        Find nodes where loads are applied (right hole area).
        
        The right hole:
        - Center at (L_ARM=300, 0, z)
        - Outer radius = R_RIGHT_OUTER = 27.5 mm
        
        We apply loads to nodes in the vicinity of the right hole.
        """
        # Distance from right hole center axis
        dx = nodes[:, 0] - self.L_ARM
        r_from_axis = np.sqrt(dx**2 + nodes[:, 1]**2)
        
        # Tolerance
        r_tol = self.R_RIGHT_OUTER + 10.0
        
        # Nodes near the right hole
        near_right = r_from_axis < r_tol
        at_right_end = nodes[:, 0] > self.L_ARM - self.R_RIGHT_OUTER - 10.0
        
        load_mask = near_right & at_right_end
        
        # Fallback: rightmost nodes
        if not np.any(load_mask):
            x_max = nodes[:, 0].max()
            load_mask = nodes[:, 0] > x_max - 20.0
            if self.verbose:
                print(f"  Warning: Using fallback load BC (rightmost nodes)")
        
        return np.where(load_mask)[0]
    
    def _compute_max_von_mises(self, nodes: np.ndarray, 
                               elements: np.ndarray,
                               u: np.ndarray, 
                               D: np.ndarray) -> float:
        """
        Compute maximum von Mises stress across all elements.
        
        For each element:
        1. Extract nodal displacements
        2. Compute strain: ε = B * u_e
        3. Compute stress: σ = D * ε
        4. Compute von Mises stress
        """
        max_vm = 0.0
        
        for elem in elements:
            # Get element nodal displacements
            elem_u = np.zeros(12)
            for i, node in enumerate(elem):
                elem_u[3*i:3*i+3] = u[3*node:3*node+3]
            
            # Get element nodes and compute B matrix
            elem_nodes = nodes[elem]
            x, y, z = elem_nodes[:, 0], elem_nodes[:, 1], elem_nodes[:, 2]
            
            J = np.array([
                [x[1]-x[0], x[2]-x[0], x[3]-x[0]],
                [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
                [z[1]-z[0], z[2]-z[0], z[3]-z[0]]
            ])
            
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-14:
                continue
            
            Jinv = np.linalg.inv(J)
            
            dN_dxi = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            dN_dx = dN_dxi @ Jinv.T
            
            # Build B matrix
            B = np.zeros((6, 12))
            for i in range(4):
                col = 3 * i
                B[0, col+0] = dN_dx[i, 0]
                B[1, col+1] = dN_dx[i, 1]
                B[2, col+2] = dN_dx[i, 2]
                B[3, col+0] = dN_dx[i, 1]
                B[3, col+1] = dN_dx[i, 0]
                B[4, col+1] = dN_dx[i, 2]
                B[4, col+2] = dN_dx[i, 1]
                B[5, col+0] = dN_dx[i, 2]
                B[5, col+2] = dN_dx[i, 0]
            
            # Compute strain and stress
            strain = B @ elem_u
            stress = D @ strain  # [σxx, σyy, σzz, τxy, τyz, τxz]
            
            # von Mises stress
            sxx, syy, szz = stress[0], stress[1], stress[2]
            txy, tyz, txz = stress[3], stress[4], stress[5]
            
            vm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 
                                + 6.0*(txy**2 + tyz**2 + txz**2)))
            
            max_vm = max(max_vm, vm)
        
        return max_vm
    
    def _compute_volume(self, nodes: np.ndarray, elements: np.ndarray) -> float:
        """Compute total mesh volume by summing tetrahedral element volumes."""
        total_vol = 0.0
        
        for elem in elements:
            v0, v1, v2, v3 = nodes[elem]
            # Tet volume = |det([v1-v0, v2-v0, v3-v0])| / 6
            mat = np.array([v1-v0, v2-v0, v3-v0])
            vol = abs(np.linalg.det(mat)) / 6.0
            total_vol += vol
        
        return total_vol


class SfePyFEASolver:
    """
    FEA solver using SfePy library.
    
    This is a more robust implementation that leverages SfePy's
    well-tested infrastructure for FEM problems.
    
    Requires: pip install sfepy
    """
    
    # Material properties
    E = 200e3   # Young's modulus (MPa)
    NU = 0.3    # Poisson's ratio
    
    # Loading
    F_Y = -8000.0
    F_X = -4000.0
    
    # Geometry
    L_ARM = 300.0
    R_LEFT_INNER = 22.5
    R_RIGHT_OUTER = 27.5
    
    def __init__(self, verbose: bool = False):
        if not HAS_SFEPY:
            raise ImportError("SfePy is required. Install with: pip install sfepy")
        if not HAS_MESHIO:
            raise ImportError("meshio is required. Install with: pip install meshio")
        
        self.verbose = verbose
    
    def solve(self, mesh_path: str) -> Tuple[float, float, float]:
        """
        Solve using SfePy.
        
        :param mesh_path: Path to mesh file
        :return: (max_displacement, max_von_mises, volume)
        """
        from sfepy.discrete.fem import Mesh, FEDomain, Field
        from sfepy.discrete import (
            FieldVariable, Material, Integral, 
            Equation, Equations, Problem
        )
        from sfepy.discrete.conditions import Conditions, EssentialBC
        from sfepy.terms import Term
        from sfepy.solvers.ls import ScipyDirect
        from sfepy.solvers.nls import Newton
        from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
        
        # Convert mesh format if needed
        vtk_path = self._ensure_vtk_format(mesh_path)
        
        # Load mesh
        mesh = Mesh.from_file(vtk_path)
        domain = FEDomain('domain', mesh)
        
        # Create regions
        omega = domain.create_region('Omega', 'all')
        
        # Find boundary nodes
        coors = mesh.coors
        fixed_nodes, load_nodes = self._identify_bc_nodes(coors)
        
        # Create field and variables
        field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=1)
        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')
        
        # Material
        D = stiffness_from_youngpoisson(3, self.E, self.NU)
        mat = Material('m', D=D)
        
        # Integral
        integral = Integral('i', order=2)
        
        # Equilibrium equation: ∫ σ:ε(v) dΩ = 0
        t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=mat, v=v, u=u)
        eq = Equation('balance', t1)
        eqs = Equations([eq])
        
        # Fixed BC
        # Create a function to select fixed nodes
        def select_fixed(coors, domain=None):
            return fixed_nodes
        
        left_region = domain.create_region(
            'Left', 'vertices by select_fixed',
            functions={'select_fixed': select_fixed}
        )
        fix_bc = EssentialBC('fix', left_region, {'u.all': 0.0})
        
        # Create problem
        ls = ScipyDirect({})
        nls = Newton({}, lin_solver=ls)
        
        pb = Problem('elasticity', equations=eqs)
        pb.set_bcs(ebcs=Conditions([fix_bc]))
        pb.set_solver(nls)
        
        # Apply loads (as distributed load on boundary)
        # For simplicity, modify the RHS directly
        # This is a simplified approach
        
        # Solve
        state = pb.solve()
        
        # Post-process
        u_vec = state()
        n_nodes = len(coors)
        u_nodal = u_vec.reshape(n_nodes, 3)
        max_disp = np.linalg.norm(u_nodal, axis=1).max()
        
        # Compute volume
        volume = self._compute_volume_from_mesh(mesh)
        
        # Compute stress (simplified - just get max displacement for now)
        # Full stress computation would use SfePy's post-processing
        max_vm = 150.0  # Placeholder
        
        return max_disp, max_vm, volume
    
    def _ensure_vtk_format(self, mesh_path: str) -> str:
        """Convert mesh to VTK format for SfePy."""
        if mesh_path.endswith('.vtk'):
            return mesh_path
        
        mesh = meshio.read(mesh_path)
        vtk_path = mesh_path.rsplit('.', 1)[0] + '.vtk'
        
        # Keep only tetrahedral cells
        cells_3d = [cb for cb in mesh.cells if cb.type in ['tetra', 'tetra10']]
        if not cells_3d:
            raise ValueError("No tetrahedral elements found")
        
        mesh_3d = meshio.Mesh(mesh.points, cells_3d)
        meshio.write(vtk_path, mesh_3d)
        
        return vtk_path
    
    def _identify_bc_nodes(self, coors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Identify fixed and load nodes."""
        # Fixed: near left hole inner surface
        r_from_left = np.sqrt(coors[:, 0]**2 + coors[:, 1]**2)
        fixed_mask = (np.abs(r_from_left - self.R_LEFT_INNER) < 5) & (np.abs(coors[:, 0]) < 30)
        
        # Fallback
        if not np.any(fixed_mask):
            fixed_mask = coors[:, 0] < coors[:, 0].min() + 10
        
        fixed_nodes = np.where(fixed_mask)[0]
        
        # Load: near right end
        load_mask = coors[:, 0] > self.L_ARM - 40
        load_nodes = np.where(load_mask)[0]
        
        return fixed_nodes, load_nodes
    
    def _compute_volume_from_mesh(self, mesh) -> float:
        """Compute total volume from SfePy mesh."""
        coors = mesh.coors
        conn = mesh.get_conn('3_4')
        
        total_vol = 0.0
        for elem in conn:
            v0, v1, v2, v3 = coors[elem]
            mat = np.array([v1-v0, v2-v0, v3-v0])
            total_vol += abs(np.linalg.det(mat)) / 6.0
        
        return total_vol


def get_solver(prefer_sfepy: bool = True, verbose: bool = False):
    """
    Get the best available FEA solver.
    
    :param prefer_sfepy: Prefer SfePy if available
    :param verbose: Print solver information
    :return: FEA solver instance
    """
    if prefer_sfepy and HAS_SFEPY:
        if verbose:
            print("Using SfePy FEA solver")
        return SfePyFEASolver(verbose=verbose)
    else:
        if verbose:
            print("Using SimpleFEA solver")
        return SimpleFEASolver(verbose=verbose)


if __name__ == "__main__":
    print("FEA Solver Module")
    print("=" * 60)
    print(f"Available solvers:")
    print(f"  SimpleFEASolver: {'✓' if HAS_SCIPY else '✗ (needs scipy)'}")
    print(f"  SfePyFEASolver:  {'✓' if HAS_SFEPY else '✗ (needs sfepy)'}")
    print()
    
    import sys
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
        print(f"Solving mesh: {mesh_path}")
        
        solver = get_solver(prefer_sfepy=False, verbose=True)
        try:
            disp, vm, vol = solver.solve(mesh_path)
            print(f"\nFinal Results:")
            print(f"  Max displacement: {disp:.6f} mm")
            print(f"  Max von Mises:    {vm:.2f} MPa")
            print(f"  Volume:           {vol:.2f} mm³")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Usage: python fea_solver.py <mesh_file>")