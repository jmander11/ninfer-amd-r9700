import unittest

from tools.r9700.check_a8q4_dflash_small_t_static import (
    EXACT_OCCUPANCY, EXACT_VGPR, WIDTHS, check,
)


def fixture(*, t10_dot8: int = 160, t24_vgpr: int = 99) -> str:
    blocks = []
    for tokens in WIDTHS:
        dot8 = t10_dot8 if tokens == 10 else 16 * tokens
        vgpr = t24_vgpr if tokens == 24 else EXACT_VGPR[tokens]
        symbol = f"kernel_a8q4g64_linear_dflash_small_t_kernelILj{tokens}EE_tail"
        blocks.append(
            f"; -- Begin function {symbol}\n"
            + "\tglobal_load_b64 v[0:1], v[2:3], off\n" * 4
            + "\tglobal_load_d16_b16 v0, v1, off\n" * (tokens + 1)
            + "\tv_dot8_i32_iu4 v0, v1, v2, v3\n" * dot8
            + f"; -- End function\n.amdhsa_kernel {symbol}\n"
              ".amdhsa_group_segment_fixed_size 0\n"
              ".amdhsa_private_segment_fixed_size 0\n"
              f".amdhsa_next_free_vgpr {vgpr}\n"
              ".amdhsa_wavefront_size32 1\n.end_amdhsa_kernel\n"
              f"; ScratchSize: 0\n; Occupancy: {EXACT_OCCUPANCY[tokens]}\n"
        )
    yaml = []
    for tokens in WIDTHS:
        vgpr = t24_vgpr if tokens == 24 else EXACT_VGPR[tokens]
        yaml.append(
            f".max_flat_workgroup_size: 256\n"
            f".name: kernel_a8q4g64_linear_dflash_small_t_kernelILj{tokens}EE_tail\n"
            f".vgpr_count: {vgpr}\n.vgpr_spill_count: 0\n.wavefront_size: 32\n"
        )
    return "".join(blocks + yaml)


class StaticGateTest(unittest.TestCase):
    def test_accepts_exact_flattened_width_union(self) -> None:
        self.assertEqual(set(check(fixture())), set(WIDTHS))

    def test_rejects_missing_dot8(self) -> None:
        with self.assertRaisesRegex(ValueError, "T10: expected 160"):
            check(fixture(t10_dot8=159))

    def test_rejects_resource_regression(self) -> None:
        with self.assertRaisesRegex(ValueError, "T24: resource identity mismatch"):
            check(fixture(t24_vgpr=100))

    def test_rejects_extra_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact flattened width kernel set"):
            check(fixture() + "; -- Begin function fake_a8q4g64_linear_dflash_small_t_kernelILj9EE\n")


if __name__ == "__main__":
    unittest.main()
