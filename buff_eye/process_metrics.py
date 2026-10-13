"""Read resource counts for this process only; no game process access."""
import ctypes
from ctypes import wintypes
import os


def process_metrics():
    if os.name != 'nt':
        return {'available': False}
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    user = ctypes.WinDLL('user32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel.GetCurrentProcess()
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counts = Counters(); counts.cb = ctypes.sizeof(counts)
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counts), counts.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessHandleCount.restype = wintypes.BOOL
    handles = wintypes.DWORD()
    if not kernel.GetProcessHandleCount(process, ctypes.byref(handles)):
        raise ctypes.WinError(ctypes.get_last_error())
    user.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    user.GetGuiResources.restype = wintypes.DWORD
    return {'available': True, 'working_set_bytes': counts.WorkingSetSize,
            'private_bytes': counts.PrivateUsage, 'handles': handles.value,
            'gdi_objects': user.GetGuiResources(process, 0),
            'user_objects': user.GetGuiResources(process, 1)}
