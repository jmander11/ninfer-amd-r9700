"""Negative checks over freshly compiled ISA for the pipeline's real hazards."""
from pathlib import Path
import re
import sys
import unittest

from check_a8q4_gate_up_prefetch_static import check

ASSEMBLY = Path(sys.argv.pop(1)).read_text()


class PrefetchStaticTest(unittest.TestCase):
    def test_compiled_pipeline(self):
        self.assertEqual(check(ASSEMBLY)['dependency_iterations_checked'], 2)
        self.assertTrue(check(ASSEMBLY)['terminal_group_dependencies_checked'])

    def terminal_start(self):
        head = re.search(r'^(\.LBB\d+_\d+):.*Inner Loop Header', ASSEMBLY, re.M)[1]
        return re.search(r'\bs_cbranch_scc0\s+' + re.escape(head) + r'\b', ASSEMBLY).end()

    def test_reject_missing_terminal_waits(self):
        start = self.terminal_start()
        changed = ASSEMBLY[:start] + re.sub(
            r'^\s*s_wait_loadcnt\s+0x[0-9a-f]+\s*$', '', ASSEMBLY[start:], flags=re.M)
        with self.assertRaisesRegex(ValueError, 'retired B64 dependency'):
            check(changed)

    def test_reject_missing_terminal_scale_wait(self):
        start = self.terminal_start()
        changed = ASSEMBLY[:start] + ASSEMBLY[start:].replace('\ts_wait_loadcnt 0x0\n', '', 1)
        with self.assertRaisesRegex(ValueError, 'vector operand consumed before its VMEM wait'):
            check(changed)

    def test_reject_missing_terminal_activation_wait(self):
        start = self.terminal_start()
        changed = ASSEMBLY[:start] + ASSEMBLY[start:].replace('\ts_wait_kmcnt 0x0\n', '', 1)
        with self.assertRaisesRegex(ValueError, 'terminal A8 scalar operands'):
            check(changed)

    def test_reject_early_drain(self):
        changed = ASSEMBLY.replace('\tv_dot8_i32_iu4',
                                   '\ts_wait_loadcnt 0x0\n\tv_dot8_i32_iu4', 1)
        with self.assertRaisesRegex(ValueError, 'drained'):
            check(changed)

    def test_reject_missing_predecessor_wait(self):
        changed = re.sub(r'^\s*s_wait_loadcnt\s+0x[0-9a-f]+\s*$', '', ASSEMBLY,
                         flags=re.M)
        with self.assertRaisesRegex(ValueError, 'before its VMEM wait'):
            check(changed)

    def test_reject_wrong_weight_generation(self):
        head = ASSEMBLY.index('Inner Loop Header')
        reg = re.search(r'global_load_b64 v\[(\d+):', ASSEMBLY[head:])[1]
        changed = re.sub(r'(v_dot8_i32_iu4 v\d+, s\d+, )v\d+',
                         lambda m: m[1] + 'v' + reg, ASSEMBLY, count=1)
        with self.assertRaisesRegex(ValueError, 'retired B64 dependency|wrong weight generation'):
            check(changed)

    def test_reject_resource_regression(self):
        changed = re.sub(r'(\.vgpr_count:\s*)\d+', r'\g<1>65', ASSEMBLY)
        with self.assertRaisesRegex(ValueError, 'register footprint'):
            check(changed)


if __name__ == '__main__':
    unittest.main()
