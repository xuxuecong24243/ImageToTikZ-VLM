"""Resource-limited worker around the existing XeLaTeX compiler (no shell).

This is NOT filesystem/network isolation. Public deployment still needs a sandbox.
Windows Job Objects / POSIX rlimits constrain worker + compiler memory and CPU.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from latex_compiler import CompileResult, _terminate_process_tree

_JOB = None  # Keep the Windows job handle open throughout compilation.


def _limits(memory_mb, cpu_seconds):
    global _JOB
    if os.name != 'nt':
        import resource
        memory = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        seconds = max(1, math.ceil(cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
        return 'posix_rlimit_as_cpu_fsize'
    import ctypes as c
    from ctypes import wintypes as w

    class Basic(c.Structure):
        _fields_ = [('process_time', c.c_longlong), ('job_time', c.c_longlong),
                    ('flags', w.DWORD), ('min_working', c.c_size_t), ('max_working', c.c_size_t),
                    ('active_processes', w.DWORD), ('affinity', c.c_size_t),
                    ('priority', w.DWORD), ('scheduling', w.DWORD)]

    class IO(c.Structure):
        _fields_ = [(name, c.c_ulonglong) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class Extended(c.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', c.c_size_t),
                    ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]

    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
    kernel.SetInformationJobObject.restype = w.BOOL
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.AssignProcessToJobObject.restype = w.BOOL
    kernel.GetCurrentProcess.restype = w.HANDLE
    _JOB = kernel.CreateJobObjectW(None, None)
    if not _JOB:
        raise OSError('Cannot create compiler resource job.')
    limits = Extended()
    # Job CPU, process+job memory, kill children when the worker/job handle closes.
    limits.basic.flags = 0x4 | 0x100 | 0x200 | 0x2000
    limits.basic.job_time = int(cpu_seconds * 10_000_000)
    limits.process_memory = limits.job_memory = memory_mb * 1024 * 1024
    if not kernel.SetInformationJobObject(_JOB, 9, c.byref(limits), c.sizeof(limits)):
        raise OSError('Cannot set compiler resource limits.')
    if not kernel.AssignProcessToJobObject(_JOB, kernel.GetCurrentProcess()):
        raise OSError('Cannot assign compiler worker to resource job; refusing unbounded compilation.')
    return 'windows_job_memory_and_cpu'


def compile_bounded(tex_path, *, xelatex='xelatex', timeout=60, memory_mb=512):
    """Return CompileResult; failure to apply limits stops before XeLaTeX."""
    tex = Path(tex_path).resolve()
    if not math.isfinite(timeout) or not 1 <= timeout <= 120:
        raise ValueError('Compile timeout must be in [1,120] seconds.')
    if isinstance(memory_mb, bool) or not isinstance(memory_mb, int) or not 256 <= memory_mb <= 2048:
        raise ValueError('Compiler memory limit must be in [256,2048] MB.')
    if tex.stat().st_size > 2_000_000:
        raise ValueError('Generated TeX exceeds the 2 MB compilation limit.')
    result_path = tex.parent/'worker_result.json'
    if result_path.exists():
        raise ValueError('Compiler worker output already exists; use a fresh iteration directory.')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(tex),
               str(xelatex), str(timeout), str(memory_mb)]
    env = os.environ.copy()
    for name in ('API_KEY', 'API_BASE_URL', 'MODEL_NAME'):
        env.pop(name, None)
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                  shell=False, env=env, cwd=str(tex.parent))
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs['start_new_session'] = True
    process = subprocess.Popen(command, **kwargs)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout+10)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
        stdout, stderr = process.communicate(timeout=5)
    worker_log = tex.parent/'worker.log'
    worker_log.write_text('exit={} timeout={}\n'.format(process.returncode, timed_out)
                          +stdout.decode('utf-8', errors='replace')
                          +'\n'+stderr.decode('utf-8', errors='replace'), encoding='utf-8')
    if process.returncode == 0 and result_path.exists() and not timed_out:
        data = json.loads(result_path.read_text(encoding='utf-8'))['compile']
        for key in ('pdf_path', 'log_path', 'console_path'):
            if data[key] is not None:
                data[key] = Path(data[key])
        return CompileResult(**data)
    message = 'Compiler worker failed or exceeded resource limits; inspect worker.log.'
    log = tex.with_suffix('.log')
    if not log.exists():
        log.write_text(message+'\n', encoding='utf-8')
    return CompileResult(False, process.returncode, timed_out, None, log, worker_log, message, command)


def _worker(argv):
    tex, engine, timeout, memory = argv
    method = _limits(int(memory), float(timeout))
    from latex_compiler import compile_tex
    result = compile_tex(Path(tex), xelatex=engine, timeout=float(timeout))
    output = {'compile': asdict(result), 'resource_limits': {'method': method,
              'memory_mb': int(memory), 'cpu_seconds': float(timeout), 'is_security_sandbox': False}}
    (Path(tex).parent/'worker_result.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--worker':
        raise SystemExit('Internal compiler worker; use auto_pipeline.py.')
    _worker(sys.argv[2:])
