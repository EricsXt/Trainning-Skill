#!/usr/bin/env python3
"""Read-only Linux container inventory. No NCCL initialization or training."""
import argparse
import ctypes
import json
from pathlib import Path
import resource
import socket
import subprocess
import time


def resolve(host, port, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
        except socket.gaierror as exc:
            if exc.errno not in {socket.EAI_AGAIN, socket.EAI_NONAME} or time.monotonic() >= deadline:
                raise
            time.sleep(min(2, max(0, deadline - time.monotonic())))


def ib_devices():
    lib = ctypes.CDLL('libibverbs.so.1', mode=ctypes.RTLD_GLOBAL, use_errno=True)
    pointer = ctypes.c_void_p
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ibv_get_device_list.restype = ctypes.POINTER(pointer)
    lib.ibv_get_device_name.argtypes = [pointer]
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_open_device.argtypes = [pointer]
    lib.ibv_open_device.restype = pointer
    lib.ibv_close_device.argtypes = [pointer]
    lib.ibv_close_device.restype = ctypes.c_int
    lib.ibv_free_device_list.argtypes = [ctypes.POINTER(pointer)]
    count = ctypes.c_int()
    devices = lib.ibv_get_device_list(ctypes.byref(count))
    if not devices:
        raise OSError(ctypes.get_errno(), 'ibv_get_device_list failed')
    result = []
    try:
        for index in range(count.value):
            device = devices[index]
            name = lib.ibv_get_device_name(device).decode()
            active = []
            for port in (Path('/sys/class/infiniband') / name / 'ports').glob('*'):
                state = port / 'state'
                if state.exists() and state.read_text().strip().startswith('4:'):
                    active.append(port.name)
            context = lib.ibv_open_device(device)
            error_number = ctypes.get_errno() if not context else None
            if context:
                lib.ibv_close_device(context)
            result.append({'name': name, 'active_ports': active, 'opened': bool(context), 'errno': error_number})
    finally:
        lib.ibv_free_device_list(devices)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-gpus', type=int)
    parser.add_argument('--coordinator-host')
    parser.add_argument('--coordinator-port', type=int, default=29500)
    parser.add_argument('--dns-timeout', type=float, default=180)
    parser.add_argument('--require-ib', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.coordinator_port <= 65535 or not 0 <= args.dns_timeout <= 600:
        parser.error('Port must be 1..65535 and DNS timeout 0..600 seconds')
    if args.expect_gpus is not None and args.expect_gpus < 1:
        parser.error('--expect-gpus must be positive')
    report = {'scope': 'Local inventory/device-open checks only; no remote TCP, NCCL or training verification', 'errors': []}
    report['hostname'] = socket.gethostname()
    report['interfaces'] = [name for _, name in socket.if_nameindex()]
    report['infiniband_device_nodes'] = sorted(p.name for p in Path('/dev/infiniband').glob('*'))
    report['memlock_limit'] = list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
        report['gpu_rows'] = [line for line in result.stdout.splitlines() if line.strip()]
        if args.expect_gpus is not None and len(report['gpu_rows']) != args.expect_gpus:
            report['errors'].append('Visible GPU count differs from --expect-gpus')
    except (OSError, subprocess.SubprocessError) as exc:
        report['gpu_error'] = type(exc).__name__
        if args.expect_gpus is not None:
            report['errors'].append('Cannot inventory requested GPUs')
    if args.coordinator_host:
        try:
            addresses = resolve(args.coordinator_host.strip('[]'), args.coordinator_port, args.dns_timeout)
            report['coordinator_addresses'] = sorted({entry[4][0] for entry in addresses})
            failures = []
            for family, kind, proto, _, address in addresses:
                try:
                    with socket.socket(family, kind, proto) as sock:
                        sock.connect(address)  # UDP local route selection; sends no payload.
                        report['local_route_address'] = sock.getsockname()[0]
                        break
                except OSError as exc:
                    failures.append(str(exc))
            if 'local_route_address' not in report:
                report['errors'].append('No route to coordinator: ' + '; '.join(failures))
        except OSError as exc:
            report['errors'].append('Coordinator resolution: ' + str(exc))
    try:
        report['ib_devices'] = ib_devices()
        if args.require_ib and not any(d['opened'] and d['active_ports'] for d in report['ib_devices']):
            report['errors'].append('No accessible IB device with an active port')
    except (OSError, AttributeError) as exc:
        report['ib_error'] = str(exc)
        if args.require_ib:
            report['errors'].append('IB userspace/device check failed')
    report['passed_requested_checks'] = not report['errors']
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed_requested_checks'] else 2)


if __name__ == '__main__':
    main()
