#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path

from tools.bench.extract_embedded_code_object import (
    DEFAULT_OBJCOPY,
    OFFLOAD_MAGIC,
    _select_device_code_object,
    extract,
)


class ExtractEmbeddedCodeObjectTest(unittest.TestCase):
    @staticmethod
    def bundle(payload: bytes, target: str = "hipv4-amdgcn-amd-amdhsa--gfx1201") -> bytes:
        identifier = target.encode()
        header_size = len(OFFLOAD_MAGIC) + 8 + 24 + len(identifier)
        offset = (header_size + 15) // 16 * 16
        return b"".join((
            OFFLOAD_MAGIC, struct.pack("<Q", 1),
            struct.pack("<QQQ", offset, len(payload), len(identifier)), identifier,
            b"\0" * (offset - header_size), payload,
        ))

    def test_selects_one_exact_gfx1201_inner_object_by_symbol(self) -> None:
        symbol = "exact_kernel_symbol"
        wanted = b"\x7fELF-device-" + symbol.encode()
        other = b"\x7fELF-unrelated"
        fatbin = self.bundle(other) + b"\0" * 7 + self.bundle(wanted)
        selected, target, offset = _select_device_code_object(fatbin, symbol)
        self.assertEqual(selected, wanted)
        self.assertIn("gfx1201", target)
        self.assertEqual(fatbin[offset:offset + len(wanted)], wanted)
        with self.assertRaisesRegex(ValueError, "found 0"):
            _select_device_code_object(fatbin, "missing")
        with self.assertRaisesRegex(ValueError, "found 2"):
            _select_device_code_object(fatbin + self.bundle(wanted), symbol)
        with self.assertRaisesRegex(ValueError, "found 0"):
            _select_device_code_object(self.bundle(wanted, target=
                "hipv4-amdgcn-amd-amdhsa--gfx12010"), symbol)

    def test_rejects_overlapping_or_empty_bundle_entries(self) -> None:
        symbol = "exact_kernel_symbol"
        payload = b"\x7fELF-device-" + symbol.encode()
        identifier = b"hipv4-amdgcn-amd-amdhsa--gfx1201"
        header_size = len(OFFLOAD_MAGIC) + 8 + 2 * 24 + 2 * len(identifier)
        offset = (header_size + 15) // 16 * 16
        overlapping = b"".join((
            OFFLOAD_MAGIC, struct.pack("<Q", 2),
            struct.pack("<QQQ", offset, len(payload), len(identifier)), identifier,
            struct.pack("<QQQ", offset + 1, len(payload) - 1, len(identifier)), identifier,
            b"\0" * (offset - header_size), payload,
        ))
        with self.assertRaisesRegex(ValueError, "found 0"):
            _select_device_code_object(overlapping, symbol)

        empty = b"".join((
            OFFLOAD_MAGIC, struct.pack("<Q", 1),
            struct.pack("<QQQ", offset, 0, len(identifier)), identifier,
            b"\0" * (offset - (len(OFFLOAD_MAGIC) + 8 + 24 + len(identifier))),
        ))
        with self.assertRaisesRegex(ValueError, "found 0"):
            _select_device_code_object(empty, symbol)

    def fixture(self, root: Path) -> tuple[Path, Path, bytes]:
        payload = b"embedded-device-code"
        executable = root / "ninfer_bench"
        executable.write_bytes(b"host-prefix" + payload + b"host-suffix")
        objcopy = root / "llvm-objcopy"
        objcopy.write_bytes(b"tool")
        return executable, objcopy, payload

    @staticmethod
    def runner(payload: bytes, *, mutate: Path | None = None, commands: list | None = None):
        def run(command, **kwargs):
            if commands is not None:
                commands.append((command, kwargs))
            Path(command[2].split("=", 1)[1]).write_bytes(payload)
            Path(command[-1]).write_bytes(b"distinct rewritten output")
            if mutate is not None:
                mutate.write_bytes(mutate.read_bytes() + b"changed")
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return run

    def test_extracts_with_distinct_output_elf_and_preserves_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            original = executable.read_bytes()
            commands = []
            output = root / "selected.hip_fatbin"
            result = extract(executable, output, objcopy=objcopy,
                             runner=self.runner(payload, commands=commands))
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(executable.read_bytes(), original)
            command, kwargs = commands[0]
            self.assertEqual(command[-2], str(executable.resolve()))
            self.assertNotEqual(Path(command[-1]), executable.resolve())
            self.assertTrue(kwargs["check"])
            self.assertEqual(result["code_object_size_bytes"], len(payload))
            self.assertEqual(result["executable_inode"], executable.stat().st_ino)
            self.assertEqual(result["executable_device"], executable.stat().st_dev)

    def test_extracts_symbol_selected_inner_device_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            symbol = "selected_kernel"
            device = b"\x7fELF-" + symbol.encode()
            fatbin = self.bundle(device)
            executable, objcopy, _ = self.fixture(root)
            executable.write_bytes(b"host-prefix" + fatbin + b"host-suffix")
            output = root / "selected.hsaco"
            result = extract(
                executable, output, objcopy=objcopy, code_symbol=symbol,
                runner=self.runner(fatbin),
            )
            self.assertEqual(output.read_bytes(), device)
            self.assertEqual(result["code_symbol"], symbol)
            self.assertIn("gfx1201", result["bundle_target"])

    def test_rejects_input_mutation_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            output = root / "selected.hip_fatbin"
            with self.assertRaisesRegex(ValueError, "changed during"):
                extract(executable, output, objcopy=objcopy,
                        runner=self.runner(payload, mutate=executable))
            self.assertFalse(os.path.lexists(output))

    def test_rejects_identical_byte_inode_replacement_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            original = executable.read_bytes()
            output = root / "selected.hip_fatbin"

            def replace(command, **kwargs):
                Path(command[2].split("=", 1)[1]).write_bytes(payload)
                Path(command[-1]).write_bytes(b"distinct rewritten output")
                replacement = root / "replacement"
                replacement.write_bytes(original)
                os.replace(replacement, executable)
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with self.assertRaisesRegex(ValueError, "changed during"):
                extract(executable, output, objcopy=objcopy, runner=replace)
            self.assertFalse(os.path.lexists(output))

    def test_rejects_same_path_inode_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "same inode"):
                extract(executable, executable.parent / "." / executable.name,
                        objcopy=objcopy, runner=self.runner(payload))
            alias = root / "alias"
            os.link(executable, alias)
            with self.assertRaisesRegex(ValueError, "same inode"):
                extract(executable, alias, objcopy=objcopy, runner=self.runner(payload))
            symlink_alias = root / "symlink-alias"
            symlink_alias.symlink_to(executable)
            with self.assertRaisesRegex(ValueError, "same inode"):
                extract(executable, symlink_alias, objcopy=objcopy,
                        runner=self.runner(payload))
            occupied = root / "occupied"
            occupied.write_bytes(b"occupied")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                extract(executable, occupied, objcopy=objcopy, runner=self.runner(payload))
            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                extract(executable, dangling, objcopy=objcopy, runner=self.runner(payload))

    def test_rejects_missing_section_or_failed_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            output = root / "selected.hip_fatbin"

            def missing(command, **kwargs):
                Path(command[-1]).write_bytes(b"distinct rewritten output")
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with self.assertRaisesRegex(ValueError, "no nonempty"):
                extract(executable, output, objcopy=objcopy, runner=missing)

            def failed(command, **kwargs):
                raise subprocess.CalledProcessError(1, command, stderr=b"bad ELF")

            with self.assertRaisesRegex(ValueError, "bad ELF"):
                extract(executable, output, objcopy=objcopy, runner=failed)
            self.assertFalse(os.path.lexists(output))

    def test_rejects_temporary_symlink_or_hardlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            output = root / "selected.hip_fatbin"

            def section_alias(command, **kwargs):
                Path(command[2].split("=", 1)[1]).symlink_to(executable)
                Path(command[-1]).write_bytes(b"distinct rewritten output")
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with self.assertRaisesRegex(ValueError, "private regular file"):
                extract(executable, output, objcopy=objcopy, runner=section_alias)

            def output_alias(command, **kwargs):
                Path(command[2].split("=", 1)[1]).write_bytes(payload)
                os.link(executable, command[-1])
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with self.assertRaisesRegex(ValueError, "private regular file"):
                extract(executable, output, objcopy=objcopy, runner=output_alias)
            self.assertFalse(os.path.lexists(output))

    def test_concurrent_output_creation_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, objcopy, payload = self.fixture(root)
            output = root / "selected.hip_fatbin"

            def occupy(command, **kwargs):
                Path(command[2].split("=", 1)[1]).write_bytes(payload)
                Path(command[-1]).write_bytes(b"distinct rewritten output")
                output.write_bytes(b"concurrent owner")
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                extract(executable, output, objcopy=objcopy, runner=occupy)
            self.assertEqual(output.read_bytes(), b"concurrent owner")

    @unittest.skipUnless(DEFAULT_OBJCOPY.is_file() and Path("/bin/true").is_file(),
                         "synthetic ELF test requires llvm-objcopy and /bin/true")
    def test_real_objcopy_extracts_exact_synthetic_section_without_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"synthetic-hip-fatbin-section\x00\x01"
            payload_path = root / "payload.bin"
            payload_path.write_bytes(payload)
            executable = root / "synthetic.elf"
            subprocess.run([
                str(DEFAULT_OBJCOPY), "--add-section", f".hip_fatbin={payload_path}",
                "/bin/true", str(executable),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            before = (executable.stat().st_dev, executable.stat().st_ino,
                      executable.read_bytes())
            output = root / "selected.hip_fatbin"
            result = extract(executable, output)
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual((executable.stat().st_dev, executable.stat().st_ino,
                              executable.read_bytes()), before)
            self.assertEqual(result["code_object_sha256"],
                             hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
