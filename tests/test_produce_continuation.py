"""The producer routes a context-reading method (the sampling baseline) without a capture."""

from wsbench.produce.methods import METHODS, Continuation, Readout
from wsbench.produce.producer import Producer
from wsbench.produce.render import Rendered


class _Backend:
    def capture(self, *a, **k):
        raise AssertionError("a context-reading method must not trigger a capture")


class _Ctx:
    name = "ctx"
    reads_context = True

    def __init__(self):
        self.layers = [42]
        self.seen = []

    def read_context(self, ids, layer):
        self.seen.append((list(ids), layer))
        return Readout(samples=[f"- {len(ids)}"])


def test_context_method_gets_the_prefix_through_the_read_token():
    m = _Ctx()
    p = Producer(backend=_Backend(), method=m)
    r = Rendered(ids=[10, 11, 12, 13, 14], tokens=list("abcde"), decoded=list("abcde"), render="x")
    rows = list(p._read_rendered("it", r, [1, -1], [42]))
    assert [(x.pos, x.token, x.readout.samples) for x in rows] == [
        (1, "b", ["- 2"]),
        (4, "e", ["- 5"]),
    ]
    assert m.seen == [([10, 11], 42), ([10, 11, 12, 13, 14], 42)]


def test_continuation_is_registered_with_the_olens_budget():
    c = METHODS["continuation"]()
    assert isinstance(c, Continuation) and c.reads_context
    assert (c.layers, c.bullets, c.max_new_tokens, c.k) == ([42], 4, 24, 1)
