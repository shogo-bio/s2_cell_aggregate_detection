"""3D instance-based quantification of S2 cell adhesion.

Importing this package must never pull in torch or cellpose. Segmentation
adapters are imported lazily, only by the factory, only when asked for.
"""

__version__ = "0.1.0"
