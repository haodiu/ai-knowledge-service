"""Eval runner (Plan §16.4, Week 8): 20-30 golden questions run through the real graph, scored
deterministically against structural expectations, with human-graded columns left blank.

CLAUDE.md is explicit about the division of labor: "you build the runner, the human owns the gold
answers." This package builds the runner, the schema for a human to author gold answers into, and
a small synthetic corpus to develop/demo against -- it does not author real gold answers for real
documents, and it never grades answer correctness or groundedness itself (see app/eval/runner.py).

    python -m app.eval.cli run --questions app/eval/golden/questions.json \
        --out ./eval_reports/2026-09-23
"""
