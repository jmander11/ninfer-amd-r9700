"""Focused tests for the Qwen3.8 corpus-to-draft-ranking provenance tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.convert.qwen3_8_27b_r9700 import build_draft_ranking as ranking


class DraftRankingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def corpus(
        self, name: str, text: str, *, tokens: int | None = None,
        extra: dict[str, object] | None = None,
    ) -> Path:
        ids = self.root / f"{name}.ids"
        payload = text.encode("ascii")
        ids.write_bytes(payload)
        manifest: dict[str, object] = {
            "artifact_type": "ninfer_ppl_corpus",
            "schema_version": 2,
            "model_id": ranking.MODEL_ID,
            "add_special_tokens": False,
            "chat_template": False,
            "tokens": len(text.split()) if tokens is None else tokens,
            "ids_sha256": hashlib.sha256(payload).hexdigest(),
            "source": name,
        }
        if extra:
            manifest.update(extra)
        ids.with_name(ids.stem + ".manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        return ids

    def test_combines_exact_counts_and_writes_one_little_endian_row(self) -> None:
        first = self.corpus("first", "3 1 3\n42")
        second = self.corpus("second", "42\t3 9")
        output = self.root / "ranking.i64"
        written, sidecar = ranking.build_ranking([first, second], output)

        self.assertEqual(written, output)
        self.assertEqual(output.stat().st_size, ranking.RANKING_BYTES)
        payload = output.read_bytes()
        self.assertEqual(struct.unpack_from("<q", payload, 1 * 8)[0], 1)
        self.assertEqual(struct.unpack_from("<q", payload, 3 * 8)[0], 3)
        self.assertEqual(struct.unpack_from("<q", payload, 9 * 8)[0], 1)
        self.assertEqual(struct.unpack_from("<q", payload, 42 * 8)[0], 2)
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["total_tokens"], 7)
        self.assertEqual(report["distinct_token_ids"], 4)
        self.assertEqual(report["ranking_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertIn("converter-owned", report["special_ids"])
        provenance = ranking.validate_ranking_provenance(output)
        self.assertEqual(provenance.ranking_sha256, report["ranking_sha256"])
        self.assertEqual(provenance.total_tokens, 7)
        self.assertEqual(len(provenance.corpora), 2)

    def test_rejects_tiled_throughput_corpus(self) -> None:
        corpus = self.corpus(
            "bench_corpus", "1 2 1 2",
            extra={"note": "tiled and repeated only to fill throughput length"},
        )
        with self.assertRaisesRegex(ValueError, "bias draft frequency counts"):
            ranking.load_corpus(corpus)

    def test_rejects_non_decimal_and_out_of_domain_ids(self) -> None:
        malformed = self.corpus("malformed", "1 -2 3")
        with self.assertRaisesRegex(ValueError, "ASCII decimal"):
            ranking.load_corpus(malformed)
        out_of_domain = self.corpus("outside", str(ranking.TOKENIZER_ID_COUNT))
        with self.assertRaisesRegex(ValueError, "outside"):
            ranking.load_corpus(out_of_domain)

    def test_rejects_manifest_count_digest_and_duplicate_input(self) -> None:
        bad_count = self.corpus("bad_count", "1 2", tokens=3)
        with self.assertRaisesRegex(ValueError, "manifest declares 3"):
            ranking.load_corpus(bad_count)
        bad_digest = self.corpus("bad_digest", "1 2", extra={"ids_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "does not match manifest"):
            ranking.load_corpus(bad_digest)
        valid = ranking.load_corpus(self.corpus("valid", "1 2"))
        with self.assertRaisesRegex(ValueError, "more than once"):
            ranking.combine_corpora([valid, valid])

    def test_rejects_conflicting_qwen_identity(self) -> None:
        corpus = self.corpus(
            "conflict", "1 2",
            extra={
                "model_id": "different-model",
                "tokenizer_model_id": ranking.TOKENIZER_MODEL_ID,
            },
        )
        with self.assertRaisesRegex(ValueError, "model_id must be"):
            ranking.load_corpus(corpus)

    def test_rejects_non_ppl_or_template_affected_provenance(self) -> None:
        wrong_type = self.corpus(
            "wrong_type", "1 2", extra={"artifact_type": "ninfer_bench_corpus"}
        )
        with self.assertRaisesRegex(ValueError, "artifact_type must be"):
            ranking.load_corpus(wrong_type)
        wrong_schema = self.corpus(
            "wrong_schema", "1 2", extra={"schema_version": 1}
        )
        with self.assertRaisesRegex(ValueError, "schema_version must be"):
            ranking.load_corpus(wrong_schema)
        templated = self.corpus("templated", "1 2", extra={"chat_template": True})
        with self.assertRaisesRegex(ValueError, "chat_template must be"):
            ranking.load_corpus(templated)

    def test_provenance_rejects_changed_row_and_claims(self) -> None:
        corpus = self.corpus("valid", "1 2 2")
        output = self.root / "ranking.i64"
        _, sidecar = ranking.build_ranking([corpus], output)

        changed = bytearray(output.read_bytes())
        struct.pack_into("<q", changed, 1 * 8, 9)
        output.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "ranking_sha256"):
            ranking.validate_ranking_provenance(output)

        report = json.loads(sidecar.read_text(encoding="utf-8"))
        report["ranking_sha256"] = hashlib.sha256(changed).hexdigest()
        report["total_tokens"] = 10
        sidecar.write_text(json.dumps(report) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frequency row does not match"):
            ranking.validate_ranking_provenance(output)


if __name__ == "__main__":
    unittest.main()
