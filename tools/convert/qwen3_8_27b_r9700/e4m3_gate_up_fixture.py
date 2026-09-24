"""Write the bounded layer-0 gate/up rowwise-E4M3 qualification fixture."""

from __future__ import annotations

from pathlib import Path

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, Payload, TensorSpec

from . import e4m3_inventory


FIXTURE_IDENTITY = ArtifactIdentity(
    e4m3_inventory.MODEL_ID, "r9700-f8e4m3-row-gate-up-fixture"
)
FIXTURE_SPEC = TensorSpec(
    name="text/layers/0/mlp/gate_up",
    shape=(34816, 5120),
    format=e4m3_inventory.F8E4M3_ROW_F32S,
    layout=e4m3_inventory.ROW_SCALED_LAYOUT,
)


def write_gate_up_fixture(path: str | Path, encoded_payload: Payload) -> Path:
    """Atomically write exactly one pre-encoded gate/up qualification object.

    The caller owns BF16-source encoding through ``encode_e4m3_rowwise``. An
    iterable payload keeps this single-object boundary streamable; the artifact
    writer enforces the exact full-shape byte count and refuses replacement.
    """

    output = Path(path)
    with ArtifactWriter(output, FIXTURE_IDENTITY, (FIXTURE_SPEC,)) as writer:
        writer.write(FIXTURE_SPEC.name, encoded_payload)
    return output


__all__ = ["FIXTURE_IDENTITY", "FIXTURE_SPEC", "write_gate_up_fixture"]
