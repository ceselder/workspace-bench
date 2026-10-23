"""Silent bridge: a novel two-hop question answered in ONE WORD ("ANSWER IN ONE WORD, DO NOT SAY
ANYTHING ELSE."); the model gets it right and never names the bridge entity it had to work out.
Can a lens name the bridge where the reply opens, or at the last prompt token?"""

from wsbench.celeste import celeste_family
from wsbench.registry import register

SPEC = register(celeste_family("celeste_silent_bridge", "Silent bridge", calls_per_arm="≈ 0.8k"))
