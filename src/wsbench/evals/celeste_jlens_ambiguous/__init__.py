"""J-lens intermediates: a question forces the model through a MULTI-WORD named entity it never
writes; read where the J-lens surfaces fragments of it (not copied from the text). Does a lens
name the whole entity?"""

from wsbench.celeste import celeste_family
from wsbench.registry import register

SPEC = register(
    celeste_family("celeste_jlens_intermediates", "J-lens intermediates", calls_per_arm="≈ 0.3k")
)
