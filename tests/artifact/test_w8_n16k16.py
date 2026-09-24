import struct

import pytest

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encoded_size, transcode_w8_n16k16, w8_n16k16_geometry


@pytest.mark.parametrize("shape", [(16, 1), (32, 129), (48, 5120)])
def test_exact_code_and_scale_permutation(shape, tmp_path):
    n, k = shape
    geometry = w8_n16k16_geometry(shape)
    groups = geometry.groups_per_row
    payload = bytearray(geometry.payload_bytes)
    for row in range(n):
        for column in range(geometry.k_pad):
            payload[row * geometry.k_pad + column] = ((row * 17 + column * 29) % 255 - 127) & 255 if column < k else 0
        for group in range(groups):
            # Positive finite FP16 words, plus exact signed-zero preservation.
            word = (0x8000 if (row + group) % 13 == 0 else 0x0400 + (row * 31 + group) % 0x7000)
            struct.pack_into("<H", payload, geometry.scale_offset + 2 * (row * groups + group), word)
    tiled = b"".join(transcode_w8_n16k16(payload, shape))
    # Independent scalar address oracle, not inverse-transform-only coverage.
    for row in range(n):
        for column in range(geometry.k_pad):
            index = (((row // 16 * groups + column // 32) * 2 + column % 32 // 16) * 16 + row % 16) * 16 + column % 16
            assert tiled[index] == payload[row * geometry.k_pad + column]
        for group in range(groups):
            target = geometry.scale_offset + ((row // 16 * groups + group) * 16 + row % 16) * 2
            source = geometry.scale_offset + (row * groups + group) * 2
            assert tiled[target:target + 2] == payload[source:source + 2]
    assert b"".join(transcode_w8_n16k16(tiled, shape, inverse=True)) == payload
    layout = "r9700-w8g32-n16-k16-v1"
    assert encoded_size(layout, "W8G32_F16S", shape) == len(payload)
    path = tmp_path / "head.ninfer"
    with ArtifactWriter(path, ArtifactIdentity("test", "layout"), [TensorSpec("head", shape, "W8G32_F16S", layout)]) as writer:
        writer.write("head", tiled)
    with Artifact.open(path) as artifact:
        view = artifact.payload("head")
        assert view == tiled
        view.release()


def test_invalid_geometry_format_and_extent():
    with pytest.raises(ValueError):
        w8_n16k16_geometry((17, 128))
    with pytest.raises(ValueError):
        encoded_size("r9700-w8g32-n16-k16-v1", "Q4G64_F16S", (16, 128))
    with pytest.raises(ValueError):
        b"".join(transcode_w8_n16k16(b"short", (16, 128)))
