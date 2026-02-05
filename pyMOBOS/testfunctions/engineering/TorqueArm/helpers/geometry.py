"""
Torque Arm Geometry Builder using Gmsh

Based on STEP file analysis:
- Bosses: 40mm deep (fixed), extending from z=-20 to z=+20
- Arm: b1 deep, centered at z=0 (e.g., z=-17.5 to z=+17.5 for b1=35)
- I-beam web: t1 deep, centered at z=0 (e.g., z=-10 to z=+10 for t1=20)
- I-beam pockets: cut from z=t1/2 to z=b1/2 (and -b1/2 to -t1/2)
- Pocket Y range: follows taper, leaves t2 thick flanges at top and bottom
- Pocket X range: from D1/2+5 to 265

The I-beam cross-section (section A-A in drawing):
- Total width: b1 (Z direction)
- Total height: varies with taper (Y direction)
- Web: thickness t1 (Z direction), centered
- Flanges: thickness t2 (Y direction), at top and bottom
"""

import numpy as np
import tempfile
from typing import Tuple, Optional, List

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False


class TorqueArmGeometry:
    """
    Parametric geometry builder for the torque arm.
    """
    
    # Fixed geometry constants (mm)
    L_ARM = 300.0           # Center-to-center distance
    D_LEFT_INNER = 45.0     # Left hole inner diameter
    D_RIGHT_INNER = 30.0    # Right hole inner diameter
    D_RIGHT_OUTER = 55.0    # Right boss outer diameter
    BOSS_DEPTH = 40.0       # Boss depth (fixed at 40mm)
    
    def __init__(self, mesh_size: float = 4.0, verbose: bool = False):
        if not HAS_GMSH:
            raise ImportError("Gmsh is required. Install with: pip install gmsh")
        self.mesh_size = mesh_size
        self.verbose = verbose
        
    def build_and_mesh(self, alpha: float, b1: float, D1: float,
                       h: float, t1: float, t2: float,
                       output_path: Optional[str] = None) -> Tuple[str, float]:
        """
        Build the torque arm geometry and generate mesh.
        
        Parameters:
            alpha: Taper angle in degrees
            b1: Arm width (Z direction)
            D1: Left boss outer diameter
            h: Height at right end (Y direction)
            t1: Web thickness (Z direction)
            t2: Flange thickness (Y direction)
        """
        if not self._validate_geometry(alpha, b1, D1, h, t1, t2):
            raise ValueError("Invalid geometry parameters")
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.msh')
        
        gmsh.initialize()
        if not self.verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        
        try:
            gmsh.model.add("torque_arm")
            volume = self._build_geometry(alpha, b1, D1, h, t1, t2)
            
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size * 1.5)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            
            gmsh.model.mesh.generate(3)
            gmsh.write(output_path)
            
            return output_path, volume
            
        finally:
            gmsh.finalize()
    
    def _validate_geometry(self, alpha, b1, D1, h, t1, t2) -> bool:
        if alpha < 0 or alpha > 15:
            return False
        if b1 <= 0 or D1 <= self.D_LEFT_INNER or h <= 0:
            return False
        if t1 <= 0 or t2 <= 0:
            return False
        if h <= 2 * t2 + 1:
            return False
        if b1 <= t1 + 1:
            return False
        if D1 <= self.D_LEFT_INNER + 5:
            return False
        return True
    
    def _build_geometry(self, alpha: float, b1: float, D1: float,
                        h: float, t1: float, t2: float) -> float:
        """Build the geometry with boss rings pre-cut and pockets cut before fusing."""
        occ = gmsh.model.occ
        
        alpha_rad = np.radians(alpha)
        R_left = D1 / 2
        R_left_inner = self.D_LEFT_INNER / 2
        R_right = self.D_RIGHT_OUTER / 2
        R_right_inner = self.D_RIGHT_INNER / 2
        
        y_top_left = self.L_ARM * np.tan(alpha_rad) + h / 2
        y_top_right = h / 2
        
        z_boss_start = -self.BOSS_DEPTH / 2
        z_arm_start = -b1 / 2
        
        # ================================================================
        # STEP 1: Create left boss ring (with hole pre-cut)
        # ================================================================
        left_boss_outer = occ.addCylinder(0, 0, z_boss_start, 0, 0, self.BOSS_DEPTH, R_left)
        left_boss_inner = occ.addCylinder(0, 0, z_boss_start - 1, 0, 0, self.BOSS_DEPTH + 2, R_left_inner)
        occ.synchronize()
        
        cut_result = occ.cut([(3, left_boss_outer)], [(3, left_boss_inner)],
                            removeObject=True, removeTool=True)
        occ.synchronize()
        left_boss = self._get_entity(cut_result[0], 3, "left boss ring")
        
        # ================================================================
        # STEP 2: Create right boss ring (with hole pre-cut)
        # ================================================================
        right_boss_outer = occ.addCylinder(self.L_ARM, 0, z_boss_start, 0, 0, self.BOSS_DEPTH, R_right)
        right_boss_inner = occ.addCylinder(self.L_ARM, 0, z_boss_start - 1, 0, 0, self.BOSS_DEPTH + 2, R_right_inner)
        occ.synchronize()
        
        cut_result = occ.cut([(3, right_boss_outer)], [(3, right_boss_inner)],
                            removeObject=True, removeTool=True)
        occ.synchronize()
        right_boss = self._get_entity(cut_result[0], 3, "right boss ring")
        
        # ================================================================
        # STEP 3: Create arm (extruded tapered profile)
        # ================================================================
        left_disk = occ.addDisk(0, 0, z_arm_start, R_left, R_left)
        right_disk = occ.addDisk(self.L_ARM, 0, z_arm_start, R_right, R_right)
        
        p_tl = occ.addPoint(0, y_top_left, z_arm_start)
        p_tr = occ.addPoint(self.L_ARM, y_top_right, z_arm_start)
        p_br = occ.addPoint(self.L_ARM, -y_top_right, z_arm_start)
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
        profile_surf = self._get_entity(fuse_result[0], 2, "profile")
        
        extrude_result = occ.extrude([(2, profile_surf)], 0, 0, b1)
        occ.synchronize()
        
        arm_solid = None
        for dim, tag in extrude_result:
            if dim == 3:
                arm_solid = tag
                break
        
        # ================================================================
        # STEP 4: Cut I-beam pockets from arm BEFORE fusing
        # ================================================================
        cut_x_start = R_left + 5
        cut_x_end = 265.0
        pocket_z_depth = (b1 - t1) / 2
        
        if pocket_z_depth > 0.5 and cut_x_end > cut_x_start + 5:
            cut_length = cut_x_end - cut_x_start
            
            t_start = cut_x_start / self.L_ARM
            y_outer_start = y_top_left + (y_top_right - y_top_left) * t_start
            
            t_end = cut_x_end / self.L_ARM
            y_outer_end = y_top_left + (y_top_right - y_top_left) * t_end
            
            y_web_top = ((y_outer_start - t2) + (y_outer_end - t2)) / 2
            y_web_bot = ((-y_outer_start + t2) + (-y_outer_end + t2)) / 2
            web_height = y_web_top - y_web_bot
            
            if web_height > 1.0:
                # Front pocket
                front_pocket = occ.addBox(
                    cut_x_start, y_web_bot, t1/2,
                    cut_length, web_height, pocket_z_depth + 1
                )
                occ.synchronize()
                
                cut_result = occ.cut([(3, arm_solid)], [(3, front_pocket)],
                                    removeObject=True, removeTool=True)
                occ.synchronize()
                arm_solid = self._get_entity(cut_result[0], 3, "front pocket")
                
                # Back pocket
                back_pocket = occ.addBox(
                    cut_x_start, y_web_bot, -b1/2 - 1,
                    cut_length, web_height, pocket_z_depth + 1
                )
                occ.synchronize()
                
                cut_result = occ.cut([(3, arm_solid)], [(3, back_pocket)],
                                    removeObject=True, removeTool=True)
                occ.synchronize()
                arm_solid = self._get_entity(cut_result[0], 3, "back pocket")
        
        # ================================================================
        # STEP 5: Fuse arm with boss rings
        # ================================================================
        fuse_result = occ.fuse([(3, arm_solid)], [(3, left_boss), (3, right_boss)],
                               removeObject=True, removeTool=True)
        occ.synchronize()
        main_solid = self._get_entity(fuse_result[0], 3, "final fuse")
        
        # ================================================================
        # FINAL: Compute volume
        # ================================================================
        occ.synchronize()
        volume = occ.getMass(3, main_solid)
        
        gmsh.model.addPhysicalGroup(3, [main_solid], 1)
        gmsh.model.setPhysicalName(3, 1, "Volume")
        
        return volume
    
    def _get_entity(self, result: List[Tuple[int, int]], dim: int, name: str) -> int:
        """Extract entity tag from boolean result."""
        entities = [tag for d, tag in result if d == dim]
        if not entities:
            raise RuntimeError(f"Boolean {name} failed")
        if len(entities) > 1:
            if dim == 3:
                measures = [(gmsh.model.occ.getMass(3, tag), tag) for tag in entities]
                measures.sort(reverse=True)
                return measures[0][1]
        return entities[0]


class TorqueArmGeometrySimple:
    """Simplified geometry without I-beam cutouts."""
    
    L_ARM = 300.0
    D_LEFT_INNER = 45.0
    D_RIGHT_INNER = 30.0
    D_RIGHT_OUTER = 55.0
    BOSS_DEPTH = 40.0
    
    def __init__(self, mesh_size: float = 4.0, verbose: bool = False):
        if not HAS_GMSH:
            raise ImportError("Gmsh is required")
        self.mesh_size = mesh_size
        self.verbose = verbose
    
    def build_and_mesh(self, alpha: float, b1: float, D1: float,
                       h: float, t1: float, t2: float,
                       output_path: Optional[str] = None) -> Tuple[str, float]:
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.msh')
        
        gmsh.initialize()
        if not self.verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        
        try:
            gmsh.model.add("torque_arm_simple")
            occ = gmsh.model.occ
            
            alpha_rad = np.radians(alpha)
            R_left = D1 / 2
            R_left_inner = self.D_LEFT_INNER / 2
            R_right = self.D_RIGHT_OUTER / 2
            R_right_inner = self.D_RIGHT_INNER / 2
            
            y_top_left = self.L_ARM * np.tan(alpha_rad) + h / 2
            y_top_right = h / 2
            z_arm_start = -b1 / 2
            z_boss_start = -self.BOSS_DEPTH / 2
            
            # Arm profile
            left_disk = occ.addDisk(0, 0, z_arm_start, R_left, R_left)
            right_disk = occ.addDisk(self.L_ARM, 0, z_arm_start, R_right, R_right)
            
            p_tl = occ.addPoint(0, y_top_left, z_arm_start)
            p_tr = occ.addPoint(self.L_ARM, y_top_right, z_arm_start)
            p_br = occ.addPoint(self.L_ARM, -y_top_right, z_arm_start)
            p_bl = occ.addPoint(0, -y_top_left, z_arm_start)
            
            l1 = occ.addLine(p_tl, p_tr)
            l2 = occ.addLine(p_tr, p_br)
            l3 = occ.addLine(p_br, p_bl)
            l4 = occ.addLine(p_bl, p_tl)
            
            trap_loop = occ.addCurveLoop([l1, l2, l3, l4])
            trap_surf = occ.addPlaneSurface([trap_loop])
            occ.synchronize()
            
            fuse_result = occ.fuse([(2, left_disk)], [(2, right_disk), (2, trap_surf)])
            occ.synchronize()
            profile = [tag for d, tag in fuse_result[0] if d == 2][0]
            
            extrude_result = occ.extrude([(2, profile)], 0, 0, b1)
            occ.synchronize()
            arm_solid = [tag for d, tag in extrude_result if d == 3][0]
            
            # Bosses
            left_boss = occ.addCylinder(0, 0, z_boss_start, 0, 0, self.BOSS_DEPTH, R_left)
            right_boss = occ.addCylinder(self.L_ARM, 0, z_boss_start, 0, 0, self.BOSS_DEPTH, R_right)
            occ.synchronize()
            
            fuse_result = occ.fuse([(3, arm_solid)], [(3, left_boss), (3, right_boss)])
            occ.synchronize()
            main_solid = [tag for d, tag in fuse_result[0] if d == 3][0]
            
            # Holes
            left_hole = occ.addCylinder(0, 0, z_boss_start - 1, 0, 0, self.BOSS_DEPTH + 2, R_left_inner)
            occ.synchronize()
            cut_result = occ.cut([(3, main_solid)], [(3, left_hole)])
            occ.synchronize()
            main_solid = [tag for d, tag in cut_result[0] if d == 3][0]
            
            right_hole = occ.addCylinder(self.L_ARM, 0, z_boss_start - 1, 0, 0, self.BOSS_DEPTH + 2, R_right_inner)
            occ.synchronize()
            cut_result = occ.cut([(3, main_solid)], [(3, right_hole)])
            occ.synchronize()
            main_solid = [tag for d, tag in cut_result[0] if d == 3][0]
            
            occ.synchronize()
            volume = occ.getMass(3, main_solid)
            
            gmsh.model.addPhysicalGroup(3, [main_solid], 1)
            
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size * 1.5)
            gmsh.model.mesh.generate(3)
            gmsh.write(output_path)
            
            return output_path, volume
            
        finally:
            gmsh.finalize()


if __name__ == "__main__":
    print("Torque Arm Geometry Test")
    
    if not HAS_GMSH:
        print("ERROR: Gmsh not installed")
        exit(1)
    
    # Test with parameters matching STEP file
    geo = TorqueArmGeometry(mesh_size=5.0, verbose=True)
    
    alpha = 4.0   # degrees
    b1 = 35.0     # mm
    D1 = 70.0     # mm
    h = 25.0      # mm
    t1 = 20.0     # mm
    t2 = 5.0      # mm
    
    try:
        mesh_path, volume = geo.build_and_mesh(alpha, b1, D1, h, t1, t2, "test.msh")
        print(f"Volume: {volume:.2f} mm³")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()