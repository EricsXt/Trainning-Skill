#!/usr/bin/env python3
"""Read-only arithmetic checks for a pure data-parallel training plan."""
import argparse
import json
from pathlib import Path


def check(plan):
    def integer(name, minimum=1):
        value = plan[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
        return value

    if plan.get('parallelism') != 'data':
        raise ValueError('Only pure data parallelism is supported; calculate hybrid DP topology explicitly')
    nodes, cards = integer('nodes'), integer('gpus_per_node')
    replicas = integer('data_parallel_replicas')
    if replicas != nodes * cards:
        raise ValueError('Pure data parallel replicas must equal nodes * gpus_per_node')
    micro = integer('per_device_microbatch')
    accumulation = integer('gradient_accumulation_steps')
    batch = integer('global_batch_size')
    if batch != replicas * micro * accumulation:
        raise ValueError('global_batch_size != per_device_microbatch * DP replicas * gradient accumulation')
    samples, epochs, interval = integer('usable_samples'), integer('epochs'), integer('save_every_epochs')
    if type(plan.get('drop_last')) is not bool:
        raise ValueError('drop_last must be an explicit boolean')
    if not plan['drop_last']:
        raise ValueError('Partial batches/padding depend on the real sampler; this checker requires drop_last=true')
    steps = samples // batch
    if steps == 0:
        raise ValueError('No complete global optimizer batch fits in an epoch')
    warmup = integer('warmup_steps', 0)
    if warmup >= steps * epochs:
        raise ValueError('warmup_steps must be smaller than total optimizer steps')
    initialization = plan.get('initialization')
    if initialization not in {'base', 'weights-only', 'full-resume'}:
        raise ValueError('initialization must be base, weights-only, or full-resume')
    warnings = []
    if initialization == 'full-resume':
        warnings.append('Resume contract and data cursor require separate native checkpoint validation')
    saved_epochs = sorted(set(range(interval, epochs + 1, interval)) | {epochs})
    return {
        'arithmetic_verified': True,
        'total_gpus': nodes * cards,
        'global_batch_size': batch,
        'optimizer_steps_per_epoch': steps,
        'dropped_samples_per_epoch': samples % batch,
        'total_optimizer_steps': steps * epochs,
        'save_at_completed_updates': [e * steps for e in saved_epochs],
        'warmup_sample_exposure': warmup * batch,
        'warnings': warnings,
        'physical_training_verified': False,
        'scope': 'Pure data parallel, complete optimizer batches; verify accumulation and tail handling in the actual sampler',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    try:
        result = check(json.loads(args.plan.read_text()))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps({'arithmetic_verified': False, 'error': str(exc)}))
        raise SystemExit(2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
