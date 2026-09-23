"""End-to-end turn scenarios (Plan §16.3), Week 7.

These run the whole graph via `tests.unit.graph.harness.Harness` (fakes only, no DB/broker) --
they satisfy Plan §16.3's *content* (numbered scenarios below), not CLAUDE.md's Commands section
description of `tests/e2e/` as "full docker-compose stack" (that is Week 8 scope, directory not
created until then per CLAUDE.md). Promoting these to run against the real stack is Week 8 work;
until then this is graph-level, and intentionally as fast as tests/unit.

Scenario -> file:
  #8  tool 404-unauthorized / 404-not-found indistinguishable
      -> test_tool_unauthorized_or_not_found.py
  #9  unknown/write tool proposal makes zero outbound calls
      -> test_unknown_or_write_tool_rejected.py
  #10 invalid generator output twice -> safe fallback
      -> test_generator_invalid_output_twice_is_safe_fallback.py
  #12 MAX_GENERATIVE_LLM_CALLS holds with a tool in the turn
      -> test_turn_never_exceeds_generative_calls_with_tool_in_play.py
"""
