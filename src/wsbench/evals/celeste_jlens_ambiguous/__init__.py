"""J-lens ambiguous: a question forces the model through a MULTI-WORD entity it never writes; at
the read site the J-lens top 10 catches fragments of it but cannot name it. Can a lens say the
whole entity?"""

from wsbench.celeste import celeste_family
from wsbench.evals.celeste_jlens_ambiguous.prompts import FAMILY_RULE_JLA, PROMPT_VERSION
from wsbench.registry import register

SPEC = register(
    celeste_family(
        "celeste_jlens_ambiguous",
        "J-lens ambiguous",
        calls_per_arm="≈ 0.2k",
        family_rule=FAMILY_RULE_JLA,
        prompt_version=PROMPT_VERSION,
    )
)
