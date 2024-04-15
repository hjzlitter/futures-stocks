from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
from ovito.pipeline import *
import numpy as np

import ovito.pipeline.ReferenceConfigurationModifier
ws = WignerSeitzAnalysisModifier(
    per_type_occupancies = True,
    affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference)
ws.reference.load("dump.noaddFe")
