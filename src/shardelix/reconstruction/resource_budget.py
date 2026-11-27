"""Adaptive memory estimates, separated from archive and decoded-pixel limits."""
from __future__ import annotations
from dataclasses import dataclass
import os
import sys
from pathlib import Path

MIB=1024**2

def available_memory():
    estimates=[]
    if sys.platform=="win32":
        try:
            import ctypes as C
            class MemoryStatus(C.Structure):
                _fields_=[("length",C.c_uint32),("load",C.c_uint32)]+[(key,C.c_uint64) for key in
                    ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
            status=MemoryStatus();status.length=C.sizeof(status)
            if C.windll.kernel32.GlobalMemoryStatusEx(C.byref(status)):
                estimates.append(int(status.avail_phys))
        except (OSError,AttributeError):pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                estimates.append(int(line.split()[1])*1024);break
    except (OSError,ValueError,IndexError):pass
    try:estimates.append(os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
    except (OSError,ValueError,AttributeError):pass
    for cap,used in (("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes","/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit=int(Path(cap).read_text());usage=int(Path(used).read_text())
            if 0<limit<2**60:estimates.append(max(0,limit-usage))
        except (OSError,ValueError):pass
    return min((v for v in estimates if v>=0),default=None)


@dataclass(frozen=True)
class ResourceBudget:
    memory_bytes: int
    source: str
    available_bytes: int|None
    browser: bool=False

    @property
    def working_pixels(self):return max(1,self.memory_bytes//18)
    @property
    def canvas_pixels(self):return max(1,self.memory_bytes//24)
    @property
    def arc_records(self):return max(1024,min(2**30,self.memory_bytes//(3*48)))
    @property
    def vote_entries(self):return max(1024,min(2**30,self.memory_bytes//(3*72)))
    @property
    def packed_bytes(self):return max(1,self.memory_bytes//3)
    @property
    def gap_pair_budget(self):return max(16,self.memory_bytes//(128*1024))
    def describe(self):
        return {"source":self.source,"memory_budget_mib":round(self.memory_bytes/MIB,1),
                "available_memory_mib":round(self.available_bytes/MIB,1) if self.available_bytes is not None else None,
                "working_pixels":self.working_pixels,"canvas_pixels":self.canvas_pixels,
                "arc_records_per_shard":self.arc_records,"vote_entries":self.vote_entries,
                "gap_pair_budget":self.gap_pair_budget}


def resource_budget(*,browser=False,override_mib=None):
    if browser:
        available=None;default=256*MIB;source="browser-worker-budget"
    else:
        available=available_memory()
        default=int(.55*available) if available is not None else 512*MIB
        source="available-host-memory" if available is not None else "portable-fallback"
    raw=override_mib if override_mib is not None else os.getenv("SHARDELIX_MEMORY_MIB")
    if raw is not None:
        try:value=int(raw)*MIB
        except (TypeError,ValueError):raise ValueError("SHARDELIX_MEMORY_MIB must be an integer >= 32") from None
        if value<32*MIB:raise ValueError("SHARDELIX_MEMORY_MIB must be >= 32")
        default=min(value,default) if available is not None or browser else value
        source+="; explicit lower budget"
    if default<32*MIB:
        raise MemoryError("Insufficient available memory for scan reconstruction (need at least 32 MiB headroom)")
    return ResourceBudget(default,source,available,browser)
