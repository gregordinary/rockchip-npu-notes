#!/usr/bin/env python3
"""ro-pagemap.py PID -- one-shot physical-placement readout of a LIVE process.

The surviving suspects for this board's per-process wall spread -- the same configuration
read 8-13% apart on consecutive processes -- are what the allocator hands the process: where
its pages land physically (cache congruence in the physically indexed L2/L3) and how
contiguous they are. Neither is visible to a /proc snapshot of the
BOARD -- MemAvailable and buddyinfo are closed as predictors -- so this reads the PROCESS.

Requires CAP_SYS_ADMIN (run under sudo): without it /proc/PID/pagemap returns PFN 0 for
every entry and every statistic below is computed over zeros. That failure is SILENT and
looks exactly like a machine with one physical page, so `pfn_zero_frac` is emitted first
and a reader must check it is 0 before quoting anything else.

WHAT IS SAMPLED, AND WHY IT IS SAMPLED IN WINDOWS. Contiguity is a run-length property, so
a strided sample cannot see it -- consecutive virtual pages have to be read consecutively.
But reading the whole map of a 13 GB resident set is 27 MB of pagemap under mmap_lock,
against a workload whose own page faults need that lock. So each mapping is sampled as
evenly spaced WINDOWS of consecutive pages (512 by default, one 2 MB span): run lengths are
exact inside a window, the lock is taken and released once per window, and the total read is
bounded by --windows regardless of how large the process is.

THE METRICS

  l2color_cv   coefficient of variation of the 16-bin histogram of (PFN & 15). The A76's
               512 KB 8-way L2 is PIPT with 1024 sets of 64 B, so physical bits [15:12] --
               the four bits above the 4 KB page offset -- choose which quarter of the sets
               a page can occupy. A uniform draw gives cv -> 0; a skewed one concentrates
               the working set into fewer sets and evicts earlier. This is the congruence
               suspect, stated as a number.
  l3color_cv   the same over (PFN & 63), which is the wider index the 3 MB shared L3 uses.
  contig_frac  fraction of adjacent sampled page PAIRS whose PFNs differ by exactly +1 --
               physical contiguity, which is what DRAM row locality and the walker's own
               caching turn on. Deliberately independent of where the mapping starts: an
               allocation can be perfectly contiguous and still be aligned anywhere.
  mean_run     mean length in pages of those contiguous runs, capped by the window size, so
               a fully contiguous mapping reads mean_run = win_pages and a fully scattered
               one reads 1.
  vapa16       fraction of present pages with (PFN - VPN) % 16 == 0. Separate from
               contiguity on purpose: PTE_CONT needs virtual AND physical 64 KB alignment
               together, so a mapping can be perfectly contiguous and still ineligible.
               Read it as the alignment lottery, NOT as proof the kernel set PTE_CONT --
               it will not have, for anonymous memory: this board has
               CONFIG_TRANSPARENT_HUGEPAGE off, so anon folios are order 0 and the
               contiguous bit is never a candidate there whatever the addresses say.
  gib_regions  distinct 1 GiB physical regions (PFN >> 18) the sample touched, i.e. how far
               the allocation is scattered across DRAM.

DEMONSTRATED DYNAMIC RANGE. On an idle RK1 after the harness's `drop_caches` +
`compact_memory` reset, contiguity is essentially perfect and the colours essentially
uniform, so these columns are near-constant BY CONSTRUCTION and a flat column there is a
statement about the board, not a working instrument. Before any null is read off them,
check they move on the contrast `ro-selftest.sh` runs: allocating against a page cache
filled and NOT compacted drives contig_frac and mean_run down and l2color_cv up, which is
the positive control this readout needs before any flat column is read as a null.

Per class (`file` = a regular file such as the GGUF, `accel` = a rocket BO mapped through
/dev/accel, `anon` = anonymous, `other`), because the classes have different mechanisms and
an aggregate over them would average a page-cache property together with an allocator one.

Emits ONE `k=v` line on stdout, empty on any failure, so a harness can drop it with [ -s ].
"""
import os
import struct
import sys
import time

PAGE = 4096
PRESENT = 1 << 63
SWAPPED = 1 << 62
PFN_MASK = (1 << 55) - 1


def parse_args(argv):
    a = {"pid": None, "min_mb": 64, "windows": 48, "win_pages": 512, "label": ""}
    if len(argv) < 2:
        sys.exit(2)
    a["pid"] = int(argv[1])
    for i in range(2, len(argv), 2):
        k = argv[i].lstrip("-").replace("-", "_")
        if k in a and i + 1 < len(argv):
            a[k] = int(argv[i + 1]) if k != "label" else argv[i + 1]
    return a


def classify(path):
    if not path:
        return "anon"
    if path.startswith("/dev/accel") or path.startswith("/dev/dri"):
        return "accel"
    if path.startswith("[") or path.startswith("/memfd") or path.startswith("/dev/"):
        return "other"
    if path.startswith("/"):
        return "file"
    return "other"


def read_maps(pid, min_bytes):
    out = []
    with open("/proc/%d/maps" % pid) as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) < 5:
                continue
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if hi - lo < min_bytes:
                continue
            path = parts[5].strip() if len(parts) > 5 else ""
            out.append((lo, hi, classify(path)))
    return out


class Acc:
    def __init__(self):
        self.n = 0            # sampled entries
        self.present = 0
        self.zero_pfn = 0
        self.l2 = [0] * 16
        self.l3 = [0] * 64
        self.pairs = 0        # adjacent present pairs examined
        self.adj = 0          # of those, physically consecutive
        self.runs = 0         # contiguous runs closed
        self.run_pages = 0
        self.vapa = 0
        self.gib = set()

    def window(self, first_vpn, pfns):
        """pfns: PFN or None per CONSECUTIVE virtual page, starting at virtual page first_vpn."""
        self.n += len(pfns)
        run = 0
        for i, p in enumerate(pfns):
            if p is None:
                if run:
                    self.runs += 1
                    self.run_pages += run
                    run = 0
                continue
            self.present += 1
            if p == 0:
                self.zero_pfn += 1
            self.l2[p & 15] += 1
            self.l3[p & 63] += 1
            self.gib.add(p >> 18)
            if (p - (first_vpn + i)) % 16 == 0:
                self.vapa += 1
            prev = pfns[i - 1] if i else None
            if prev is not None:
                self.pairs += 1
                if p == prev + 1:
                    self.adj += 1
                    run += 1
                    continue
                if run:
                    self.runs += 1
                    self.run_pages += run
                    run = 0
            run = 1
        if run:
            self.runs += 1
            self.run_pages += run


def cv(hist):
    tot = sum(hist)
    if tot == 0:
        return 0.0
    m = tot / float(len(hist))
    var = sum((h - m) ** 2 for h in hist) / float(len(hist))
    return (var ** 0.5) / m


def main():
    a = parse_args(sys.argv)
    pid = a["pid"]
    t0 = time.time()
    accs = {}
    try:
        maps = read_maps(pid, a["min_mb"] * 1024 * 1024)
        pm = open("/proc/%d/pagemap" % pid, "rb", buffering=0)
    except (IOError, OSError, ValueError):
        return
    nmap = 0
    with pm:
        for lo, hi, cls in maps:
            npages = (hi - lo) // PAGE
            if npages < a["win_pages"]:
                continue
            nwin = min(a["windows"], npages // a["win_pages"])
            if nwin < 1:
                continue
            nmap += 1
            acc = accs.setdefault(cls, Acc())
            step = npages // nwin
            for w in range(nwin):
                # Align the window start to its own span so the c2m test can ever fire.
                first = (lo // PAGE + w * step) & ~(a["win_pages"] - 1)
                try:
                    pm.seek(first * 8)
                    buf = pm.read(a["win_pages"] * 8)
                except (IOError, OSError, ValueError):
                    return
                if len(buf) < a["win_pages"] * 8:
                    continue
                ent = struct.unpack("<%dQ" % a["win_pages"], buf)
                acc.window(first, [(e & PFN_MASK) if (e & PRESENT) else None for e in ent])
    if not accs:
        return
    tot = Acc()
    for acc in accs.values():
        tot.n += acc.n
        tot.present += acc.present
        tot.zero_pfn += acc.zero_pfn
        tot.gib |= acc.gib
        for i in range(16):
            tot.l2[i] += acc.l2[i]
        for i in range(64):
            tot.l3[i] += acc.l3[i]
        tot.pairs += acc.pairs
        tot.adj += acc.adj
        tot.runs += acc.runs
        tot.run_pages += acc.run_pages
        tot.vapa += acc.vapa
    f = []
    if a["label"]:
        f.append("who=%s" % a["label"])
    pres = max(tot.present, 1)
    f.append("pfn_zero_frac=%.4f" % (tot.zero_pfn / float(pres)))
    f.append("maps=%d" % nmap)
    f.append("sampled=%d" % tot.n)
    f.append("present_frac=%.4f" % (tot.present / float(max(tot.n, 1))))
    f.append("l2color_cv=%.4f" % cv(tot.l2))
    f.append("l3color_cv=%.4f" % cv(tot.l3))
    f.append("contig_frac=%.4f" % (tot.adj / float(max(tot.pairs, 1))))
    f.append("mean_run=%.1f" % (tot.run_pages / float(max(tot.runs, 1))))
    f.append("vapa16=%.4f" % (tot.vapa / float(pres)))
    f.append("gib_regions=%d" % len(tot.gib))
    for cls in sorted(accs):
        acc = accs[cls]
        p = max(acc.present, 1)
        f.append("%s_pg=%d" % (cls, acc.present))
        f.append("%s_l2cv=%.4f" % (cls, cv(acc.l2)))
        f.append("%s_contig=%.4f" % (cls, acc.adj / float(max(acc.pairs, 1))))
        f.append("%s_run=%.1f" % (cls, acc.run_pages / float(max(acc.runs, 1))))
        f.append("%s_gib=%d" % (cls, len(acc.gib)))
    f.append("ro_cost_ms=%d" % int((time.time() - t0) * 1000))
    sys.stdout.write("\t".join(f) + "\n")


if __name__ == "__main__":
    main()
