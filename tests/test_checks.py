"""Run with python3 -m unittest discover -s tests; no GPUs or network required."""
from pathlib import Path
import json
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'multinode-training' / 'scripts'))
from check_plan import check
from node_preflight import resolve


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / 'multinode-training/references/plan-example.json').read_text())

    def test_complete_epoch_accounting(self):
        result = check(self.plan)
        self.assertEqual(result['optimizer_steps_per_epoch'], 1345)
        self.assertEqual(result['total_optimizer_steps'], 26900)
        self.assertEqual(result['dropped_samples_per_epoch'], 838)
        self.assertEqual(result['save_at_completed_updates'], list(range(1345, 26901, 1345)))
        self.assertFalse(result['physical_training_verified'])

    def test_invalid_contracts(self):
        for delta in [{'global_batch_size': 256}, {'nodes': 0}, {'nodes': True},
                      {'data_parallel_replicas': 8}, {'gradient_accumulation_steps': 4},
                      {'parallelism': 'tensor'}, {'initialization': 'resume?'},
                      {'warmup_steps': 26900}, {'drop_last': False}, {'usable_samples': 3}]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                check({**self.plan, **delta})

    def test_gradient_accumulation(self):
        result = check({**self.plan, 'gradient_accumulation_steps': 4, 'global_batch_size': 4096})
        self.assertEqual(result['global_batch_size'], 4096)
        self.assertEqual(result['optimizer_steps_per_epoch'], 336)


class DnsTests(unittest.TestCase):
    answer = [(socket.AF_INET, socket.SOCK_DGRAM, 17, '', ('127.0.0.1', 29500))]

    def test_ready_without_retry(self):
        with patch('socket.getaddrinfo', return_value=self.answer) as mock:
            self.assertEqual(resolve('primary', 29500, 30), self.answer)
            self.assertEqual(mock.call_count, 1)

    def test_transient_errors_recover(self):
        failures = [socket.gaierror(socket.EAI_AGAIN, 'temporary'),
                    socket.gaierror(socket.EAI_NONAME, 'pending'), self.answer]
        with patch('socket.getaddrinfo', side_effect=failures), patch('time.sleep'):
            self.assertEqual(resolve('primary', 29500, 30), self.answer)

    def test_permanent_failure_has_deadline(self):
        with patch('socket.getaddrinfo', side_effect=socket.gaierror(socket.EAI_NONAME, 'missing')):
            with self.assertRaises(socket.gaierror):
                resolve('primary', 29500, 0)

    def test_other_resolver_errors_not_retried(self):
        with patch('socket.getaddrinfo', side_effect=socket.gaierror(socket.EAI_FAIL, 'bad')) as mock:
            with self.assertRaises(socket.gaierror):
                resolve('primary', 29500, 30)
            self.assertEqual(mock.call_count, 1)


if __name__ == '__main__':
    unittest.main()
