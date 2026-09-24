"""Check campaign coverage and its screen-to-finalist execution gate."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('chunk_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_attention_admission_required_before_inspecting_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            admission = Path(directory) / 'result.json'
            admission.write_text('{"status":"not_admitted"}')
            with patch.object(campaign, 'ATTENTION_ADMISSION', admission), \
                 patch.object(campaign, 'inspect_candidate_artifact') as inspect:
                with self.assertRaisesRegex(ValueError, 'not passed'):
                    campaign.inspect_inputs()
                inspect.assert_not_called()

    def test_preflight_does_not_publish_and_freeze_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'inputs.json'
            with patch.object(campaign, 'RECEIPT', receipt), \
                 patch.object(campaign, 'inspect_inputs', return_value={'files': {}}), \
                 patch.object(campaign, 'require_auto_power_profile', return_value='auto'), \
                 patch.object(campaign.sys, 'argv', ['campaign.py', 'preflight']):
                campaign.main()
                self.assertEqual(list(Path(directory).iterdir()), [])
                with patch.object(campaign.sys, 'argv', ['campaign.py', 'freeze']):
                    campaign.main()
                    before = receipt.read_bytes()
                    with self.assertRaisesRegex(ValueError, 'namespace already exists'):
                        campaign.main()
                    self.assertEqual(receipt.read_bytes(), before)

    def test_preflight_rejects_non_auto_power_without_publication(self):
        with patch.object(campaign, 'inspect_inputs', return_value={'files': {}}), \
             patch.object(campaign, 'require_auto_power_profile', side_effect=ValueError('not auto')), \
             patch.object(campaign, 'durable_create_json') as publish:
            with self.assertRaisesRegex(ValueError, 'not auto'):
                campaign.preflight()
            publish.assert_not_called()

    def test_changed_artifact_receipt_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'inputs.json'
            campaign.durable_create_json(receipt, {'files': {}, 'artifacts': ['old'] * 3})
            with patch.object(campaign, 'RECEIPT', receipt), \
                 patch.object(campaign, 'inspect_candidate_artifact', return_value='changed'), \
                 patch.object(campaign.sys, 'argv', ['campaign.py', 'screens']), \
                 patch.object(campaign, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'artifact or conversion receipt changed'):
                    campaign.main()
                run.assert_not_called()

    def test_screen_execution_covers_exact_twelve_profiles(self):
        with patch.object(campaign, 'run') as run, patch.object(campaign, 'require_idle_gpu'):
            campaign.execute('screen', (1024, 2048, 4096, 8192))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 12)
        actual = set()
        for command in commands:
            value = lambda key: command[command.index(key) + 1]
            actual.add((value('--weights'), value('--expected-kv-value-group'),
                        value('--expected-xattention-profile')))
            self.assertEqual(value('--concurrency'), '1')
            self.assertEqual(value('--prefill-prompt'), '8192')
            self.assertEqual([command[i + 1] for i, item in enumerate(command)
                              if item == '--prefill-chunk'], ['1024', '2048', '4096', '8192'])
            self.assertIn('--resume', command)
            self.assertEqual('--require-fp8-hybrid' in command,
                             'four-role' in value('--weights'))
        self.assertEqual(len(actual), 12)

    def test_occupied_gpu_blocks_measurement(self):
        with patch.object(campaign, 'run') as run, \
             patch.object(campaign, 'require_idle_gpu', side_effect=ValueError('already resident')):
            with self.assertRaisesRegex(ValueError, 'already resident'):
                campaign.execute('screen', (1024, 2048, 4096, 8192))
            run.assert_not_called()

    def test_invalid_screening_cannot_launch_finalist(self):
        with patch.object(campaign.sys, 'argv', ['campaign.py', 'finalists']), \
             patch.object(campaign, 'verify_inputs'), \
             patch.object(campaign.os.path, 'lexists', return_value=True), \
             patch.object(campaign, 'validate_screening_record', side_effect=ValueError('invalid screen')), \
             patch.object(campaign, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'invalid screen'):
                campaign.main()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
