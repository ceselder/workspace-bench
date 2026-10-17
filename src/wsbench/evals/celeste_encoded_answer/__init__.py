"""Encoded answer: the model computes an answer but must write it transformed (letter count, first
letter, list position, ...); can a lens name the entity at the end of the prompt and at the end of
the model's own encoded output?"""

from wsbench.celeste import celeste_family
from wsbench.registry import register

SPEC = register(
    celeste_family(
        "celeste_encoded_answer", "Encoded answer", calls_per_arm="≈ 0.4k", forbidden=True
    )
)
