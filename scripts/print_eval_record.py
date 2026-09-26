"""Print the committed eval snapshot in the same layout as ``python -m rag_lab eval``.

CI uses this instead of calling MiniLM or a language model. The file is
``eval/record.json``, captured from a full local harness run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RECORD = Path(__file__).resolve().parents[1] / "eval" / "record.json"


def main() -> int:
    """Load ``eval/record.json`` and print Recall@K plus generated scores."""
    payload = json.loads(_RECORD.read_text(encoding="utf-8"))
    retrieval = payload["retrieval"]
    generated = payload["generated"]
    rows = retrieval["results"]
    lines = [
        f"Eval queries={len(rows)} k={retrieval['k']}",
        f"recall_at_k={retrieval['recall_at_k']:.3f}",
        f"extractive_answer_accuracy={retrieval['extractive_answer_accuracy']:.3f}",
        f"generated_key_fact={generated['generated_key_fact']:.3f}",
        f"citation_complete={generated['citation_complete']:.3f}",
        f"groundedness={generated['groundedness']:.3f}",
        f"conflict_handling={generated['conflict_handling']:.3f}",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['query_id']}  recall={row['recall_at_k']:.0f}  "
            f"accurate={int(row['answer_correct'])}  {row['retrieval_label']}"
        )
        cited = row["cited"]
        lines.append(
            f"  {cited['doc_name']}  v{cited['version']}  "
            f"{cited['status']}  {cited['section']}"
        )
        if row.get("note"):
            lines.append(f"  {row['note']}")
    sys.stdout.write("\n".join(lines).rstrip() + "\n")
    required = (
        retrieval["recall_at_k"],
        retrieval["extractive_answer_accuracy"],
        generated["generated_key_fact"],
        generated["citation_complete"],
        generated["groundedness"],
        generated["conflict_handling"],
    )
    if any(value != 1.0 for value in required):
        sys.stderr.write("error: eval/record.json is not a perfect scoreboard\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
