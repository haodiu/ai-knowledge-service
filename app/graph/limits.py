"""Hard limits of the online graph (Plan §8.4, CLAUDE.md invariant #7).

Module constants on purpose, not settings: these are not knobs to tune away. Changing one is a
code change with a review, and `tests/unit/graph/test_limits.py` pins the values.
"""
MAX_RETRIEVAL_ATTEMPTS = 2  # first retrieval + at most one rewrite
MAX_TOOL_CALLS = 1
MAX_GENERATIVE_LLM_CALLS = 4  # every attempt counts: first calls, retries and repairs alike
MAX_RETRIEVED_CHUNKS = 8
GRAPH_TIMEOUT_SECONDS = 30

MAX_EVIDENCE_CHARS = 12_000  # context budget for the evidence block (whole chunks only)
MAX_QUESTION_CHARS = 2_000
GRAPH_RECURSION_LIMIT = 25  # last-resort backstop; the routing bounds bind long before this

MAX_HISTORY_TURNS = 4  # Plan §11.5: "tối đa 4 turns"
MAX_HISTORY_CHARS_PER_TURN = 500  # keeps stale conversations from crowding out current evidence
