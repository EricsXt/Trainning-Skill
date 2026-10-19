"""Exercise shell examples with temporary files and a fake platform/runtime."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='training skill ')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.env = {**os.environ}
        for key in ['LOCAL_RANK', 'TRAINING_ENV_FILE']:
            self.env.pop(key, None)

    def run_script(self, script, *args):
        return subprocess.run(['bash', str(EXAMPLES / script), *args], env=self.env,
                              text=True, capture_output=True, timeout=15)

    def test_shell_syntax(self):
        for script in EXAMPLES.glob('*.sh'):
            with self.subTest(script=script.name):
                result = subprocess.run(['bash', '-n', str(script)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_only_uses_real_plan(self):
        trainer = self.work / 'train.py'
        trainer.write_text('raise RuntimeError("Must not execute training during config check")\n')
        self.env.update(TRAINING_ROOT=str(self.work), TRAIN_PYTHON=sys.executable,
                        TRAIN_SCRIPT=str(trainer),
                        PLAN_PATH=str(ROOT / 'multinode-training/references/plan-example.json'))
        result = self.run_script('preflight.sh', '--config')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('26900', result.stdout)

    def test_torchrun_arguments_and_no_nested_launch(self):
        runtime = self.work / 'fake-python'
        runtime.write_text('#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if '-m' in args and 'torch.distributed.run' in args:
 Path(os.environ['CAPTURE']).write_text(json.dumps(args)); raise SystemExit(0)
if args and args[0].endswith('node_preflight.py'):
 raise SystemExit(0)  # Mock allocation inventory; no real GPUs touched.
os.execv(sys.executable, [sys.executable, *args])
''')
        runtime.chmod(0o700)
        trainer = self.work / 'train.py'
        trainer.write_text('')
        capture = self.work / 'capture.json'
        self.env.update(TRAINING_ROOT=str(self.work), TRAIN_PYTHON=str(runtime),
                        TRAIN_SCRIPT=str(trainer), CAPTURE=str(capture), NNODES='2', GPUS_PER_NODE='8',
                        NODE_RANK='1', MASTER_ADDR='current-primary', MASTER_PORT='29500', REQUIRE_IB='false',
                        PLAN_PATH=str(ROOT / 'multinode-training/references/plan-example.json'))
        result = self.run_script('torchrun_multinode.sh', '--config', '/shared/path with spaces/config.yaml')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(capture.read_text())
        self.assertIn('--nnodes=2', args)
        self.assertIn('--nproc-per-node=8', args)
        self.assertEqual(args[-2:], ['--config', '/shared/path with spaces/config.yaml'])
        self.env['NNODES'] = '3'
        self.assertNotEqual(self.run_script('torchrun_multinode.sh').returncode, 0)
        self.env['NNODES'] = '2'
        self.env['LOCAL_RANK'] = '0'
        self.assertNotEqual(self.run_script('torchrun_multinode.sh').returncode, 0)

    def adapter(self):
        adapter = self.work / 'adapter'
        adapter.write_text('#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
verb=sys.argv[1]
with Path(os.environ['TRACE']).open('a') as f:f.write(verb+'\\n')
if verb=='submit':
 if os.environ.get('FAIL_SUBMIT'):raise SystemExit(3)
 print(json.dumps({'job_id':'new-id'}))
elif verb=='detail':
 print(json.dumps({'job_id':'new-id','status':'Inqueue','configuration_verified':not bool(os.environ.get('FAIL_DETAIL'))}))
elif verb=='stop':print(json.dumps({'job_id':sys.argv[2],'stopped':True}))
''')
        adapter.chmod(0o700)
        request = self.work / 'request.json'
        request.write_text('{}')
        trace = self.work / 'trace'
        self.env.update(PLATFORM_ADAPTER=str(adapter), REQUEST_FILE=str(request), OLD_JOB_ID='old-id',
                        TRACE=str(trace), CHECK_PYTHON=sys.executable)
        return trace

    def test_replace_order(self):
        trace = self.adapter()
        result = self.run_script('replace_job.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace.read_text().splitlines(), ['submit', 'detail', 'stop'])

    def test_failed_submit_preserves_old_job(self):
        trace = self.adapter()
        self.env['FAIL_SUBMIT'] = '1'
        self.assertNotEqual(self.run_script('replace_job.sh').returncode, 0)
        self.assertEqual(trace.read_text().splitlines(), ['submit'])

    def test_failed_readback_preserves_old_job(self):
        trace = self.adapter()
        self.env['FAIL_DETAIL'] = '1'
        self.assertNotEqual(self.run_script('replace_job.sh').returncode, 0)
        self.assertEqual(trace.read_text().splitlines(), ['submit', 'detail'])

    def test_same_id_never_stopped(self):
        trace = self.adapter()
        self.env['OLD_JOB_ID'] = 'new-id'
        self.assertNotEqual(self.run_script('replace_job.sh').returncode, 0)
        self.assertEqual(trace.read_text().splitlines(), ['submit'])

    def test_monitor_once(self):
        log = self.work / 'train.log'
        log.write_text('Step 20: loss=0.1\n')
        result = self.run_script('watch_logs.sh', '--once', str(log))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('loss=0.1', result.stdout)

    @unittest.skipUnless(shutil.which('flock'), 'flock is required for lock test')
    def test_duplicate_monitor_rejected(self):
        log = self.work / 'train.log'
        log.write_text('Step 20: loss=0.1\n')
        lock = self.work / 'watch.lock'
        self.env['MONITOR_LOCK'] = str(lock)
        import fcntl
        with lock.open('w') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script('watch_logs.sh', '--follow', str(log))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('already running', result.stderr)


if __name__ == '__main__':
    unittest.main()
