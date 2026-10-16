"""Withheld answer: the model solves a riddle or describes a thing it is told never to name; can a
lens name it inside the model's own evasive reply?"""

from wsbench.celeste import celeste_family
from wsbench.registry import register

SPEC = register(
    celeste_family("celeste_withheld_answer", "Withheld answer", calls_per_arm="≈ 0.5k")
)
