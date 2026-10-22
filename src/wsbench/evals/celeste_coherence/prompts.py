"""Prompts of the celeste_coherence bullet judge; printed verbatim in the family README."""

from wsbench.llm import schema_block

PROMPT_VERSION = "coherence-0to3-2026-10-22"
CHUNK = 20  # bullets per call

SYSTEM = """You grade short text fragments for COHERENCE ONLY: is each fragment readable text that
conveys some content or claim? You do NOT grade truth, plausibility, relevance, importance,
completeness, or style.

Context. The fragments are single bullet points written by machine "lenses" that try to put a
language model's internal state into words. They are often deliberately SHORT FRAGMENTS: many
start mid-sentence (e.g. ", essentially, create the table" or "and I was developed by Alibaba
Cloud") and many are cut off at the end because the writer hit a length limit. A fragment that
starts or ends mid-sentence is still fully coherent if the words that are there read as natural
text and you can tell what it is saying. Markdown such as **bold**, *italics* or `code` is fine
and is not a flaw.

Anchored scale (choose exactly one integer per fragment):
0 = NOT LANGUAGE. Random words or word pieces, gibberish, mixed-script soup, or a list of
unrelated tokens. Nothing can be read as a statement.
1 = BROKEN. Made of real words but does not parse into any statement: scrambled word order;
severe repetition of a word or phrase (e.g. "the the the the", or the same phrase repeated
several times); a fragment cut in the middle of words at both ends so that no clause survives;
or a mid-stream switch to an unrelated script or language so the pieces do not form one thought.
2 = READABLE BUT FLAWED. You can tell what it is saying, but it is awkward, partly garbled, has
a grammar error, an intrusive foreign word, a stray repetition, or a truncation that costs real
meaning.
3 = FLUENT. Well-formed natural text (a fragment or a sentence) whose content is clear, with no
garbling. Bland, generic content still scores 3 if it reads cleanly. Very specific content
(names, numbers, dates, code identifiers, technical jargon) also scores 3 if it reads cleanly.

Rules:
- Never reward or penalize truth, plausibility, specificity, relevance, or how interesting the
  content is.
- Never penalize a fragment for starting mid-sentence, lacking a subject, or stopping before the
  sentence is finished, as long as what is there reads cleanly.
- END TRUNCATION IS NOT A FLAW. Every fragment was cut at a fixed token budget, so ending
  mid-sentence, mid-question or even mid-word (e.g. "...While our training data span") says
  nothing about its coherence. Score only the text that is present: if everything before the
  cut reads cleanly, the score is 3.
- Chat-template debris such as "<think>", "</think>", "<br/>", "\\n" or a leading "_" or "/" is
  not a coherence flaw; ignore it and score the words.
- Text entirely in one non-English language that reads as fluent text in that language is
  coherent (3). A mid-text switch to another language or script that breaks the thought is a
  flaw (1 if the thought is lost, 2 if most of it survives).
- Code or shell commands count as text: coherent if well-formed.
- Repetition of a phrase inside an otherwise normal sentence is a 2; a fragment that is mostly
  repetition is a 1.

Return one grade per fragment, in input order: {"grades": [{"i": 0, "score": 3, "why": "clean
fragment"}, {"i": 1, "score": 1, "why": "phrase repeated 5 times"}]}. "why" is at most 8
words."""

USER = "Grade these {n} fragments.\n\n{fragments}"

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "USER": USER}

SCHEMA = schema_block(
    "coherence_grades",
    {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["i", "score", "why"],
                "properties": {
                    "i": {"type": "integer"},
                    "score": {"type": "integer"},
                    "why": {"type": "string"},
                },
            },
        }
    },
    ["grades"],
)


def render_user(fragments: list[str]) -> str:
    import json

    body = json.dumps([{"i": i, "text": t} for i, t in enumerate(fragments)], ensure_ascii=False)
    return USER.replace("{n}", str(len(fragments))).replace("{fragments}", body)
