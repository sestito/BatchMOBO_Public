from .geometry import TorqueArmGeometry, TorqueArmGeometrySimple, HAS_GMSH
from .fea_solver import SimpleFEASolver, HAS_MESHIO, HAS_SCIPY

# Optional SfePy solver
try:
    from .fea_solver import SfePyFEASolver, HAS_SFEPY
except ImportError:
    SfePyFEASolver = None
    HAS_SFEPY = False

__all__ = [
    'TorqueArmGeometry',
    'TorqueArmGeometrySimple',
    'SimpleFEASolver',
    'SfePyFEASolver',
    'HAS_GMSH',
    'HAS_MESHIO',
    'HAS_SCIPY',
    'HAS_SFEPY'
]