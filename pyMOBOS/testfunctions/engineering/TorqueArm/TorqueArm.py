"""
Torque Arm Multi-Objective Optimization Test Function

A real FEA-based implementation for multi-objective optimization of a torque arm,
replacing the original MATLAB + ANSYS workflow with Python + Gmsh + custom FEA.

This follows the same class interface as WeldedBeam.py for compatibility
with optimization frameworks.

Geometry Description:
---------------------
The torque arm connects two cylindrical bosses with a tapered arm:
- Left boss: Hollow cylinder (variable outer dia D1, inner dia 45mm)
- Right boss: Hollow cylinder (outer dia 55mm, inner dia 30mm)
- Arm: Tapered I-beam cross-section connecting the bosses
- Center-to-center distance: 300mm

Design Variables:
-----------------
x1 (alpha): Taper angle [3.0, 4.5] degrees
x2 (b1): Overall width [25.0, 35.0] mm
x3 (D1): Left outer diameter [90.0, 120.0] mm
x4 (h): Height at right end [20.0, 30.0] mm
x5 (t1): Rib/web thickness [12.0, 22.0] mm
x6 (t2): Flange thickness [8.0, 12.0] mm

Objectives (minimize):
---------------------
f1: Maximum displacement [mm]
f2: Total volume [mm³]

Constraint:
-----------
g1: Maximum von Mises stress <= 190 MPa

Usage:
------
    from pyMOBOS.testfunctions.engineering.TorqueArm import TorqueArm
    import numpy as np
    
    problem = TorqueArm()
    x = np.array([[4.20, 34.05, 93.55, 29.11, 18.11, 8.58]])
    result = problem(x)  # Returns [displacement, volume]

Dependencies:
-------------
    pip install numpy scipy gmsh meshio

References:
-----------
[1] Park, H.S. and Dang, X.P. (2010). "Structural Optimization Based on CAD/CAE
    Integration and Metamodeling Techniques." CAD Computer Aided Design.

[2] Sestito, J.M., Harris, T.A.L., and Wang, Y. "Scalable Pareto Quality
    Metrics for Multi-Objective Bayesian Optimization." J. Mechanical Design.
"""

import numpy as np
import warnings
import tempfile
import os
import shutil
from typing import Tuple, Optional

from .helpers import (
    TorqueArmGeometry, 
    TorqueArmGeometrySimple,
    SimpleFEASolver,
    HAS_GMSH, 
    HAS_MESHIO
)


class TorqueArm:
    """
    Torque Arm Design Multi-Objective Optimization Problem
    
    Uses real finite element analysis to evaluate designs.
    
    Attributes:
        INFEASIBLE_VALUE: Value assigned to infeasible designs (default: 1e10)
        sigma_max: Maximum allowable stress in MPa (default: 190.0)
        
    The INFEASIBLE_VALUE is assigned to both objectives when:
    - Geometry creation fails (invalid parameters)
    - Mesh generation fails
    - FEA solution fails
    """
    
    name = "TorqueArm"
    number_of_parameters = 6
    number_of_objectives = 2
    
    # Material properties
    E = 200e3       # Young's modulus (MPa)
    nu = 0.3        # Poisson's ratio
    
    # Loading
    F_vertical = -8000.0    # N (downward)
    F_horizontal = -4000.0  # N (leftward)
    
    # Constraint
    sigma_max = 190.0       # Maximum allowable stress (MPa)
    
    # Infeasible design value
    INFEASIBLE_VALUE = 1e10
    
    def __init__(self, number_of_parameters: int = 6,
                 mesh_size: float = 5.0,
                 keep_mesh_files: bool = False,
                 work_dir: Optional[str] = None,
                 use_simple_geometry: bool = False,
                 verbose: bool = False):
        """
        Initialize the TorqueArm problem.
        
        :param number_of_parameters: Must be 6 (ignored if different)
        :param mesh_size: Target mesh element size in mm (smaller = finer)
        :param keep_mesh_files: If True, don't delete mesh files after solve
        :param work_dir: Working directory for mesh files (temp if None)
        :param use_simple_geometry: Use simplified geometry (no I-beam cutouts)
        :param verbose: Print detailed progress information
        """
        if number_of_parameters != 6:
            warnings.warn("TorqueArm uses exactly 6 parameters. Setting to 6.")
        
        self.mesh_size = mesh_size
        self.keep_mesh_files = keep_mesh_files
        self.work_dir = work_dir
        self.use_simple_geometry = use_simple_geometry
        self.verbose = verbose
        
        self.set_bounds()
        
        # Validate dependencies
        self._check_dependencies()
        
        # Create geometry builder
        if HAS_GMSH:
            if use_simple_geometry:
                self._geometry = TorqueArmGeometrySimple(
                    mesh_size=mesh_size, verbose=verbose
                )
            else:
                self._geometry = TorqueArmGeometry(
                    mesh_size=mesh_size, verbose=verbose
                )
        else:
            self._geometry = None
            
        # Create FEA solver
        if HAS_MESHIO:
            self._solver = SimpleFEASolver(verbose=verbose)
        else:
            self._solver = None
    
    def _check_dependencies(self):
        """Check if required packages are available."""
        missing = []
        if not HAS_GMSH:
            missing.append("gmsh")
        if not HAS_MESHIO:
            missing.append("meshio")
        
        if missing:
            raise ImportError(
                f"Missing required dependencies: {missing}. "
                f"Install with: pip install {' '.join(missing)}"
            )
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the torque arm objectives.
        
        :param x: 2D array [n_samples, 6] with columns [alpha, b1, D1, h, t1, t2]
        :return: 2D array [n_samples, 2] with columns [displacement, volume]
        
        Infeasible designs (geometry failure, constraint violation) return
        INFEASIBLE_VALUE for both objectives.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] != 6:
            raise ValueError(
                'TorqueArm requires exactly 6 parameters: '
                '[alpha, b1, D1, h, t1, t2]'
            )
        
        return self._compute_objectives(x)
    
    def set_number_of_parameters(self, number_of_parameters: int) -> None:
        """Cannot change number of parameters for TorqueArm."""
        warnings.warn("TorqueArm uses exactly 6 parameters. Cannot change.")
    
    def set_bounds(self) -> None:
        """
        Set parameter bounds based on Park & Dang (2010) and Sestito et al.
        
        Bounds:
            alpha: [3.0, 4.5] degrees - taper angle
            b1: [25.0, 35.0] mm - overall width
            D1: [90.0, 120.0] mm - left outer diameter
            h: [20.0, 30.0] mm - height at right
            t1: [12.0, 22.0] mm - rib thickness
            t2: [8.0, 12.0] mm - flange thickness
        """
        bounds = np.zeros((2, self.number_of_parameters))
        bounds[0, :] = [3.0, 25.0, 90.0, 20.0, 12.0, 8.0]    # Lower bounds
        bounds[1, :] = [4.5, 35.0, 120.0, 30.0, 22.0, 12.0]  # Upper bounds
        self.parameter_bounds = bounds
    
    def _compute_objectives(self, x: np.ndarray) -> np.ndarray:
        """Compute objectives for all samples."""
        n_samples = x.shape[0]
        output = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            alpha, b1, D1, h, t1, t2 = x[i, :]
            
            if self.verbose:
                print(f"\nEvaluating design {i+1}/{n_samples}:")
                print(f"  alpha={alpha:.2f}°, b1={b1:.2f}, D1={D1:.2f}")
                print(f"  h={h:.2f}, t1={t1:.2f}, t2={t2:.2f}")
            
            try:
                disp, vol, stress = self._evaluate_single(
                    alpha, b1, D1, h, t1, t2
                )
                
                if self.verbose:
                    print(f"  Results: disp={disp:.4f}mm, vol={vol:.1f}mm³, σ={stress:.1f}MPa")
                
                # Apply penalty if stress constraint violated
                if stress > self.sigma_max:
                    penalty = stress - self.sigma_max + 1.0
                    output[i, 0] = disp + penalty
                    output[i, 1] = vol + penalty
                    if self.verbose:
                        print(f"  Constraint violated: penalty={penalty:.2f}")
                else:
                    output[i, 0] = disp
                    output[i, 1] = vol
                    
            except Exception as e:
                # Infeasible design
                if self.verbose:
                    print(f"  FAILED: {e}")
                else:
                    warnings.warn(f"Design {i} failed: {e}")
                output[i, :] = self.INFEASIBLE_VALUE
        
        return output
    
    def _evaluate_single(self, alpha: float, b1: float, D1: float,
                        h: float, t1: float, t2: float) -> Tuple[float, float, float]:
        """
        Evaluate a single design point using FEA.
        
        :return: (max_displacement, volume, max_stress)
        :raises ValueError: If geometry is invalid
        :raises RuntimeError: If meshing or FEA fails
        """
        if self._geometry is None or self._solver is None:
            raise RuntimeError(
                "FEA dependencies not available. "
                "Install with: pip install gmsh meshio scipy"
            )
        
        # Validate geometry before attempting
        if not self._is_valid_geometry(alpha, b1, D1, h, t1, t2):
            raise ValueError("Invalid geometry parameters")
        
        # Create working directory
        if self.work_dir:
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
            cleanup = False
        else:
            work_dir = tempfile.mkdtemp(prefix="torque_arm_")
            cleanup = not self.keep_mesh_files
        
        try:
            mesh_path = os.path.join(work_dir, "mesh.msh")
            
            # Build geometry and mesh
            mesh_path, volume_cad = self._geometry.build_and_mesh(
                alpha, b1, D1, h, t1, t2, mesh_path
            )
            
            # Run FEA
            max_disp, max_stress, volume_mesh = self._solver.solve(mesh_path)
            
            # Use volume from mesh (more accurate than CAD)
            volume = volume_mesh
            
            return max_disp, volume, max_stress
            
        finally:
            if cleanup and os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _is_valid_geometry(self, alpha, b1, D1, h, t1, t2) -> bool:
        """
        Check if geometry parameters produce a physically valid design.
        
        Note: This only checks physical validity, not optimization bounds.
        Parameters outside bounds will still work for geometry creation.
        
        Validates:
        - All parameters positive
        - I-beam is physically possible (flanges fit in height, web fits in width)
        - D1 larger than inner hole (45mm)
        - Reasonable taper angle
        """
        # Basic positivity
        if any(v <= 0 for v in [alpha, b1, D1, h, t1, t2]):
            return False
        
        # Reasonable taper angle (0-15 degrees)
        if alpha > 15:
            return False
        
        # I-beam validity
        if h <= 2 * t2 + 1:  # Flanges must fit with some web space
            return False
        if b1 <= t1 + 1:     # Web must fit with some flange overhang
            return False
        
        # D1 must be larger than left inner hole (45mm)
        if D1 <= 45.0 + 5:
            return False
        
        return True
    
    def is_within_bounds(self, x: np.ndarray) -> bool:
        """
        Check if parameters are within the optimization bounds.
        
        :param x: 1D array [6] with parameter values
        :return: True if within bounds, False otherwise
        """
        if x.ndim == 2:
            x = x.flatten()
        
        lb = self.parameter_bounds[0]
        ub = self.parameter_bounds[1]
        
        for val, lo, hi in zip(x, lb, ub):
            if val < lo or val > hi:
                return False
        return True
    
    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate constraint violations.
        
        The stress constraint is: σ_max ≤ 190 MPa
        
        :param x: 2D array [n_samples, 6] with parameter values
        :return: 1D array where g ≤ 0 indicates feasibility
        
        Returns σ - σ_max, so negative values are feasible.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        g = np.zeros(n_samples)
        
        for i in range(n_samples):
            alpha, b1, D1, h, t1, t2 = x[i, :]
            
            try:
                _, _, stress = self._evaluate_single(alpha, b1, D1, h, t1, t2)
                g[i] = stress - self.sigma_max
            except:
                g[i] = self.INFEASIBLE_VALUE  # Infeasible
        
        return g
    
    def get_stress(self, x: np.ndarray) -> np.ndarray:
        """
        Get maximum von Mises stress for designs.
        
        Useful for constraint analysis and visualization.
        
        :param x: 2D array [n_samples, 6]
        :return: 1D array of stress values (MPa)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        stress = np.zeros(n_samples)
        
        for i in range(n_samples):
            alpha, b1, D1, h, t1, t2 = x[i, :]
            try:
                _, _, stress[i] = self._evaluate_single(alpha, b1, D1, h, t1, t2)
            except:
                stress[i] = self.INFEASIBLE_VALUE
        
        return stress
    
    def get_full_results(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all FEA results (displacement, volume, stress).
        
        :param x: 2D array [n_samples, 6]
        :return: (displacements, volumes, stresses) - each is 1D array
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        disp = np.zeros(n_samples)
        vol = np.zeros(n_samples)
        stress = np.zeros(n_samples)
        
        for i in range(n_samples):
            alpha, b1, D1, h, t1, t2 = x[i, :]
            try:
                disp[i], vol[i], stress[i] = self._evaluate_single(
                    alpha, b1, D1, h, t1, t2
                )
            except:
                disp[i] = vol[i] = stress[i] = self.INFEASIBLE_VALUE
        
        return disp, vol, stress
    
    def view_geometry(self, x: np.ndarray, 
                      output_path: Optional[str] = None,
                      show_gui: bool = True,
                      export_formats: Optional[list] = None) -> Optional[str]:
        """
        Generate and view/export the geometry for a given design.
        """
        import gmsh
        
        if x.ndim != 1 or len(x) != 6:
            if x.ndim == 2 and x.shape[0] == 1:
                x = x.flatten()
            else:
                raise ValueError("x must be a 1D array with 6 elements")
        
        alpha, b1, D1, h, t1, t2 = x
        
        if not self._is_valid_geometry(alpha, b1, D1, h, t1, t2):
            raise ValueError(f"Invalid geometry parameters")
        
        if export_formats is None and output_path is not None:
            export_formats = ['step', 'stl']
        
        gmsh.initialize()
        if not self.verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        
        try:
            gmsh.model.add("torque_arm_view")
            occ = gmsh.model.occ
            
            # Constants
            L_ARM = 300.0
            BOSS_DEPTH = 40.0
            R_left_inner = 22.5   # Ø45 / 2
            R_right = 27.5        # Ø55 / 2
            R_right_inner = 15.0  # Ø30 / 2
            
            alpha_rad = np.radians(alpha)
            R_left = D1 / 2
            
            # Heights from taper
            y_top_left = L_ARM * np.tan(alpha_rad) + h / 2
            y_top_right = h / 2
            
            # Z positions
            z_arm_start = -b1 / 2
            z_boss_start = -BOSS_DEPTH / 2
            
            if self.verbose:
                print(f"Building geometry:")
                print(f"  R_left={R_left}, R_right={R_right}")
                print(f"  y_top_left={y_top_left:.2f}, y_top_right={y_top_right:.2f}")
                print(f"  z_arm_start={z_arm_start}, z_boss_start={z_boss_start}")
            
            # ============================================================
            # STEP 1: Create left boss ring (with hole pre-cut)
            # ============================================================
            left_boss_outer = occ.addCylinder(0, 0, z_boss_start, 0, 0, BOSS_DEPTH, R_left)
            left_boss_inner = occ.addCylinder(0, 0, z_boss_start - 1, 0, 0, BOSS_DEPTH + 2, R_left_inner)
            occ.synchronize()
            
            cut_result = occ.cut([(3, left_boss_outer)], [(3, left_boss_inner)],
                                removeObject=True, removeTool=True)
            occ.synchronize()
            left_boss = self._get_entity_tag(cut_result[0], 3)
            
            if self.verbose:
                vol_left_boss = occ.getMass(3, left_boss)
                print(f"  Left boss ring: volume={vol_left_boss:.1f}")
            
            # ============================================================
            # STEP 2: Create right boss ring (with hole pre-cut)
            # ============================================================
            right_boss_outer = occ.addCylinder(L_ARM, 0, z_boss_start, 0, 0, BOSS_DEPTH, R_right)
            right_boss_inner = occ.addCylinder(L_ARM, 0, z_boss_start - 1, 0, 0, BOSS_DEPTH + 2, R_right_inner)
            occ.synchronize()
            
            cut_result = occ.cut([(3, right_boss_outer)], [(3, right_boss_inner)],
                                removeObject=True, removeTool=True)
            occ.synchronize()
            right_boss = self._get_entity_tag(cut_result[0], 3)
            
            if self.verbose:
                vol_right_boss = occ.getMass(3, right_boss)
                print(f"  Right boss ring: volume={vol_right_boss:.1f}")
            
            # ============================================================
            # STEP 3: Create arm (extruded tapered profile)
            # ============================================================
            left_disk = occ.addDisk(0, 0, z_arm_start, R_left, R_left)
            right_disk = occ.addDisk(L_ARM, 0, z_arm_start, R_right, R_right)
            
            p_tl = occ.addPoint(0, y_top_left, z_arm_start)
            p_tr = occ.addPoint(L_ARM, y_top_right, z_arm_start)
            p_br = occ.addPoint(L_ARM, -y_top_right, z_arm_start)
            p_bl = occ.addPoint(0, -y_top_left, z_arm_start)
            
            l_top = occ.addLine(p_tl, p_tr)
            l_right = occ.addLine(p_tr, p_br)
            l_bot = occ.addLine(p_br, p_bl)
            l_left = occ.addLine(p_bl, p_tl)
            
            trap_loop = occ.addCurveLoop([l_top, l_right, l_bot, l_left])
            trap_surf = occ.addPlaneSurface([trap_loop])
            occ.synchronize()
            
            fuse_result = occ.fuse([(2, left_disk)], [(2, right_disk), (2, trap_surf)],
                                   removeObject=True, removeTool=True)
            occ.synchronize()
            profile_surf = self._get_entity_tag(fuse_result[0], 2)
            
            extrude_result = occ.extrude([(2, profile_surf)], 0, 0, b1)
            occ.synchronize()
            
            arm_solid = None
            for dim, tag in extrude_result:
                if dim == 3:
                    arm_solid = tag
                    break
            
            if self.verbose:
                vol_arm = occ.getMass(3, arm_solid)
                print(f"  Arm solid: volume={vol_arm:.1f}")
            
            # ============================================================
            # STEP 4: Cut I-beam pockets from arm BEFORE fusing
            # ============================================================
            cut_x_start = R_left + 5  # D1/2 + 5
            cut_x_end = 265.0
            cut_length = cut_x_end - cut_x_start
            pocket_z_depth = (b1 - t1) / 2
            
            if self.verbose:
                print(f"  I-beam pocket: x={cut_x_start:.1f} to {cut_x_end:.1f}, z_depth={pocket_z_depth:.1f}")
            
            if pocket_z_depth > 0.5 and cut_length > 5:
                # Calculate web region (Y bounds)
                t_start = cut_x_start / L_ARM
                y_outer_start = y_top_left + (y_top_right - y_top_left) * t_start
                
                t_end = cut_x_end / L_ARM
                y_outer_end = y_top_left + (y_top_right - y_top_left) * t_end
                
                # Web region is between the flanges
                y_web_top = ((y_outer_start - t2) + (y_outer_end - t2)) / 2
                y_web_bot = ((-y_outer_start + t2) + (-y_outer_end + t2)) / 2
                web_height = y_web_top - y_web_bot
                
                if self.verbose:
                    print(f"  Web region: y={y_web_bot:.1f} to {y_web_top:.1f}, height={web_height:.1f}")
                
                if web_height > 1.0:
                    # Front pocket: from z=t1/2 to z=b1/2
                    z_front_start = t1 / 2
                    front_pocket = occ.addBox(
                        cut_x_start, y_web_bot, z_front_start,
                        cut_length, web_height, pocket_z_depth + 1
                    )
                    occ.synchronize()
                    
                    cut_result = occ.cut([(3, arm_solid)], [(3, front_pocket)],
                                        removeObject=True, removeTool=True)
                    occ.synchronize()
                    arm_solid = self._get_entity_tag(cut_result[0], 3)
                    
                    if self.verbose:
                        vol_after_front = occ.getMass(3, arm_solid)
                        print(f"  After front pocket: volume={vol_after_front:.1f}")
                    
                    # Back pocket: from z=-b1/2 to z=-t1/2
                    z_back_start = -b1/2 - 1
                    back_pocket = occ.addBox(
                        cut_x_start, y_web_bot, z_back_start,
                        cut_length, web_height, pocket_z_depth + 1
                    )
                    occ.synchronize()
                    
                    cut_result = occ.cut([(3, arm_solid)], [(3, back_pocket)],
                                        removeObject=True, removeTool=True)
                    occ.synchronize()
                    arm_solid = self._get_entity_tag(cut_result[0], 3)
                    
                    if self.verbose:
                        vol_after_back = occ.getMass(3, arm_solid)
                        print(f"  After back pocket: volume={vol_after_back:.1f}")
            
            # ============================================================
            # STEP 5: Fuse arm with boss rings
            # ============================================================
            fuse_result = occ.fuse([(3, arm_solid)], [(3, left_boss), (3, right_boss)],
                                   removeObject=True, removeTool=True)
            occ.synchronize()
            main_solid = self._get_entity_tag(fuse_result[0], 3)
            
            volume = occ.getMass(3, main_solid)
            
            if self.verbose:
                print(f"  Final volume after fuse: {volume:.1f} mm³")
            
            # Export
            exported_files = []
            if output_path and export_formats:
                for fmt in export_formats:
                    fmt = fmt.lower().strip('.')
                    file_path = f"{output_path}.{fmt}"
                    
                    if fmt in ['step', 'stp', 'brep']:
                        gmsh.write(file_path)
                        exported_files.append(file_path)
                    elif fmt in ['stl', 'vtk', 'msh']:
                        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size)
                        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size * 2)
                        gmsh.model.mesh.generate(3 if fmt != 'stl' else 2)
                        gmsh.write(file_path)
                        exported_files.append(file_path)
                
                if exported_files:
                    print(f"Exported geometry to:")
                    for f in exported_files:
                        print(f"  {f}")
                    print(f"Volume: {volume:.2f} mm³")
            
            if show_gui:
                gmsh.option.setNumber("Geometry.Surfaces", 1)
                gmsh.option.setNumber("Geometry.SurfaceType", 2)
                
                print(f"\nGeometry created:")
                print(f"  α={alpha:.2f}°, b1={b1:.2f}, D1={D1:.2f}")
                print(f"  h={h:.2f}, t1={t1:.2f}, t2={t2:.2f}")
                print(f"  Volume: {volume:.2f} mm³")
                print(f"\nOpening Gmsh GUI... (close window to continue)")
                
                gmsh.fltk.run()
            
            return exported_files[0] if exported_files else None
            
        finally:
            gmsh.finalize()
    
    def _get_entity_tag(self, result: list, dim: int) -> int:
        """Extract entity tag of given dimension from boolean result."""
        entities = [tag for d, tag in result if d == dim]
        if not entities:
            raise RuntimeError(f"Boolean operation failed - no entity of dim {dim}")
        if len(entities) > 1:
            import gmsh
            if dim == 3:
                measures = [(gmsh.model.occ.getMass(3, tag), tag) for tag in entities]
                measures.sort(reverse=True)
                return measures[0][1]
        return entities[0]