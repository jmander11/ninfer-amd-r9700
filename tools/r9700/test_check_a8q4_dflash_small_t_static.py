import unittest

from tools.r9700.check_a8q4_dflash_small_t_static import check


def fixture(*, t5_dot8: int = 80, t6_vgpr: int = 41) -> str:
    blocks = []
    for tokens, dot8, vgpr in ((4, 64, 26), (5, t5_dot8, 29), (6, 96, t6_vgpr)):
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
              "; ScratchSize: 0\n; Occupancy: 16\n"
        )
    yaml = []
    for tokens, vgpr in ((4, 26), (5, 29), (6, t6_vgpr)):
        yaml.append(
            f".max_flat_workgroup_size: 256\n"
            f".name: kernel_a8q4g64_linear_dflash_small_t_kernelILj{tokens}EE_tail\n"
            f".vgpr_count: {vgpr}\n.vgpr_spill_count: 0\n.wavefront_size: 32\n"
        )
    return "".join(blocks + yaml)


class StaticGateTest(unittest.TestCase):
    def test_accepts_exact_three_widths(self) -> None:
        self.assertEqual(check(fixture())[6]["vgpr"], 41)

    def test_rejects_missing_dot8(self) -> None:
        with self.assertRaisesRegex(ValueError, "T5: expected 80"):
            check(fixture(t5_dot8=79))

    def test_rejects_resource_regression(self) -> None:
        with self.assertRaisesRegex(ValueError, "T6: resource identity mismatch"):
            check(fixture(t6_vgpr=65))


if __name__ == "__main__":
    unittest.main()
