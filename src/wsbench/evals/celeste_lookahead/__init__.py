"""Lookahead: the model must write a fixed preamble before a one-word answer; can a lens name the
answer at the end of the prompt, at least 16 tokens before the model writes it?"""

from wsbench.celeste import celeste_family
from wsbench.registry import register

SPEC = register(
    celeste_family(
        "celeste_lookahead", "Lookahead behind a forced preamble", calls_per_arm="≈ 0.3k"
    )
)
