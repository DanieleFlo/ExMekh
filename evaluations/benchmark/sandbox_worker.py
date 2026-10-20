"""Trusted launcher: apply kernel restrictions before loading candidate Python."""

import ctypes
import ctypes.util
import json
import os
import resource
import runpy
import sys


def restrict_files(libc, paths: dict[str, int]) -> None:
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock is unavailable")
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    descriptor = libc.syscall(444, ctypes.byref(Ruleset(handled)), 8, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, access in paths.items():
            if not os.path.exists(path):
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if libc.syscall(
                    445, descriptor, 1, ctypes.byref(PathRule(access, fd)), 0
                ):
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, descriptor, 0):
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(descriptor)


def restrict_syscalls(libc, seccomp) -> None:
    # Landlock does not cover metadata changes such as chmod; block those too.
    denied = (
        "chmod",
        "fchmod",
        "fchmodat",
        "fchmodat2",
        "chown",
        "fchown",
        "lchown",
        "fchownat",
        "utime",
        "utimes",
        "futimesat",
        "utimensat",
        "setxattr",
        "lsetxattr",
        "fsetxattr",
        "removexattr",
        "lremovexattr",
        "fremovexattr",
        "mount",
        "umount2",
        "pivot_root",
        "chroot",
        "setns",
        "unshare",
        "ptrace",
        "process_vm_writev",
        "open_by_handle_at",
        "ioctl",
        "socket",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "execve",
        "execveat",
    )
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("Cannot initialize seccomp")
    try:
        for name in denied:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(context, 0x50001, number, 0):
                raise RuntimeError("Cannot restrict syscall " + name)
        for capability in range(41):
            libc.prctl(24, capability, 0, 0, 0)

        class Header(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

        class Caps(ctypes.Structure):
            _fields_ = [
                ("effective", ctypes.c_uint32),
                ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32),
            ]

        if libc.capset(ctypes.byref(Header(0x20080522, 0)), ctypes.byref((Caps * 2)())):
            raise OSError(ctypes.get_errno(), "capset")
        if seccomp.seccomp_load(context):
            raise RuntimeError("Cannot apply seccomp")
    finally:
        seccomp.seccomp_release(context)


def main() -> None:
    workspace, temporary, runtime, ready_descriptor = sys.argv[1:]
    request = json.load(sys.stdin)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        library = ctypes.util.find_library("seccomp")
        if not library:
            raise RuntimeError("libseccomp is unavailable")
        seccomp = ctypes.CDLL(library)
        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024, 384 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
        read = 1 | 4 | 8
        restrict_files(
            libc,
            {
                "/usr": read,
                runtime: read,
                workspace: read,
                temporary: (1 << 13) - 1,
                "/dev/null": 2 | 4,
                "/dev/urandom": 4,
            },
        )
        restrict_syscalls(libc, seccomp)
    except Exception as error:
        print("BENCHMARK_SANDBOX_UNAVAILABLE: " + str(error), file=sys.stderr)
        raise SystemExit(125) from error
    # Close the trusted channel before candidate code can write to descriptors.
    os.write(int(ready_descriptor), b"1")
    os.close(int(ready_descriptor))
    os.chdir(workspace)
    sys.path.insert(0, workspace)
    path = os.path.join(workspace, request["path"])
    if request["mode"] == "script":
        sys.argv = [path, *request["arguments"]]
        runpy.run_path(path, run_name="__main__")
    else:
        namespace = runpy.run_path(path)
        function = namespace[request["function"]]
        if not callable(function):
            raise TypeError("Requested target is not callable")
        if request["mode"] == "raises":
            try:
                function(*request["arguments"])
            except Exception as error:
                result = {"raised": True, "exception": type(error).__name__}
            else:
                result = {"raised": False}
            print(json.dumps(result))
            return
        result = function(*request["arguments"])
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
