import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bench import build_selected_decode_profile as builder
from tools.bench import prepare_selected_static_audit as audit
from tools.bench import verify_selected_hardware_use as verifier


class PrepareSelectedStaticAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_exact_device_symbol_and_emitted_metadata(self) -> None:
        symbol = "_Z_dense_full_score_qk_tiled_kernelILj32ELj64EE"
        raw = (
            symbol + "\0_Z___device_stub__dense_full_score_qk_tiled_kernelILj32ELj64EE\0"
        ).encode()
        self.assertEqual(
            audit.unique_symbol(raw, "dense_full_score_qk_tiled_kernelILj32ELj64EE"), symbol
        )
        with self.assertRaisesRegex(ValueError, "one selected code symbol"):
            audit.unique_symbol(
                raw + b"_Z_other_dense_full_score_qk_tiled_kernelILj32ELj64EE\0",
                "dense_full_score_qk_tiled_kernelILj32ELj64EE",
            )
        metadata = f"""amdhsa.kernels:
  - .args: []
    .name: {symbol}
    .vgpr_count: 65
    .group_segment_fixed_size: 32928
    .private_segment_fixed_size: 0
    .wavefront_size: 32
"""
        proof = audit.symbol_proof(
            symbol,
            audit.SPECS["dense_panel_qk"][2],
            "v_wmma_f32_16x16x16_bf16 v0,v1,v2,v3\n" * 64,
            metadata,
        )
        self.assertEqual(
            (proof["opcode_sites"], proof["vgpr"], proof["lds_bytes"]), (64, 65, 32928)
        )
        with self.assertRaisesRegex(ValueError, "not wave32"):
            audit.symbol_proof(symbol, "opcode", "", metadata.replace("size: 32", "size: 64"))

    def test_selected_only_writer_reopens_exact_isa_and_is_create_only(self) -> None:
        tmp_path = self.tmp_path
        names = ["q4_p2048_cta", "q4_wave32", "ordinary_fp8_qk", "dense_panel_qk"]
        symbols = {name: "_Z_" + audit.SPECS[name][1] for name in names}
        binary = tmp_path / "bench"
        binary.write_bytes(("\0".join(symbols.values()) + "\0").encode())
        route = {
            "executable": audit.snapshot(binary),
            "xattention_profile": "dense",
            "kv_value_group": 16,
            "weights_id": "r9700-q4g64-n16k16-eval",
            "terminal_selection": {"path": "/real-selection-boundary", "sha256": "a" * 64},
        }

        def extract(_source, out, **_kwargs):
            out.write_bytes(b"code")
            return {"code_object_sha256": "b" * 64}

        proofs = {
            name: {
                "code_symbol": symbols[name],
                "machine_bytes": 4,
                "machine_sha256": "c" * 64,
                "vgpr": 65,
            }
            for name in names
        }
        output = tmp_path / "audit.json"
        with (
            patch.object(audit, "selected_route", return_value=route),
            patch.object(audit, "extract", side_effect=extract),
            patch.object(
                audit,
                "inspect_symbol",
                side_effect=lambda _code, _symbol, name: dict(proofs[name]),
            ),
        ):
            value = audit.prepare(tmp_path / "selection.json", output)
            self.assertEqual(json.loads(output.read_text()), value)
            self.assertEqual(len(value["benchmark_executables"]), 1)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                audit.prepare(tmp_path / "selection.json", output)
            with (
                patch.object(verifier, "extract", side_effect=extract),
                patch.object(
                    verifier,
                    "_machine_interval",
                    return_value={"machine_bytes": 4, "machine_sha256": "c" * 64},
                ),
            ):
                verifier._selected_embedded_static(
                    binary, "dense-g16", value, value["shared_symbol_proofs"], names
                )
                value["shared_symbol_proofs"]["q4_wave32"]["vgpr"] = 64
                with self.assertRaisesRegex(ValueError, "ISA/resources differ"):
                    verifier._selected_embedded_static(
                        binary, "dense-g16", value, value["shared_symbol_proofs"], names
                    )

    def test_fresh_profile_receipt_path_reaches_selected_authority_gate(self) -> None:
        tmp_path = self.tmp_path
        selection = tmp_path / "selection.json"
        selection.write_text("{}")
        output = tmp_path / "fresh-receipt.json"
        with (
            patch.object(builder, "PROFILE_BUILD_DIR", tmp_path / "fresh-build"),
            patch.object(builder, "finalist_campaign_is_live", return_value=False),
            patch.object(
                builder, "resolve_route", side_effect=ValueError("terminal authority required")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "terminal authority required"):
                builder.build(selection, output)
            self.assertFalse(output.exists())
            output.write_text("preserve failed attempt")
            with self.assertRaisesRegex(ValueError, "occupied"):
                builder.build(selection, output)
            self.assertEqual(output.read_text(), "preserve failed attempt")


if __name__ == "__main__":
    unittest.main()
