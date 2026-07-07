#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Walk a vendor .rknn for the embedded 64-bit NPU command words and decode the PPU
# (0x6000) / PPU_RDMA (0x7000) register page (the gahingwoo capture method).
#
# A command word is NPUOP(op,value,reg) = (op<<48)|(value<<16)|reg, i.e. little-endian
# bytes [reg:u16][value:u32][target:u16] (== Mesa decode.py's '<hIh'). The .rknn embeds
# the regcmd as a contiguous run of such words; we slide a byte window, find maximal
# runs of valid (reg in map, target domain known) commands, and decode them with field
# names parsed from Mesa's registers.xml. Pooling has no weights so the run is clean.
#
# The register field map is Mesa's rocket registers.xml (MIT), vendored alongside this
# script so the harness is self-contained; override with ROCKET_REGISTERS_XML to point
# at a live Mesa checkout.
#
# Usage: decode_pool_rknn.py file1.rknn [file2.rknn ...]
import sys, struct, os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.environ.get("ROCKET_REGISTERS_XML", os.path.join(HERE, "registers.xml"))

def load_xml():
    root = ET.parse(XML).getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[:root.tag.index("}") + 1]
    regs = {}        # offset -> (domain, name, [(field, low, high, type)])
    targets = {}     # domain name -> value
    for dom in root.iter(ns + "domain"):
        dname = dom.get("name")
        for r in dom.iter(ns + "reg32"):
            off = int(r.get("offset"), 0)
            fields = []
            for b in r.iter(ns + "bitfield"):
                if b.get("pos") is not None:
                    lo = hi = int(b.get("pos"))
                else:
                    lo, hi = int(b.get("low")), int(b.get("high"))
                fields.append((b.get("name"), lo, hi, b.get("type")))
            regs[off] = (dname, r.get("name"), fields)
    for e in root.iter(ns + "enum"):
        if e.get("name") == "target":
            for v in e.iter(ns + "value"):
                targets[v.get("name")] = int(v.get("value"), 0)
    return regs, targets

REGS, TARGETS = load_xml()
TARGET_VALS = set(TARGETS.values())
# PC control ops carry no domain in the target enum: OP_ENABLE 0x81 / OP_40 0x41 /
# OP_REG_PC 0x01 / OP_NONE 0x00 all act on the PC block (regs 0x0008/0x0014/0x0000).
PC_CTRL = {0x00, 0x01, 0x41, 0x81, 0x40, 0x80}

def valid_cmd(reg, target):
    dom = target & 0xfffe
    if reg in REGS and (dom in TARGET_VALS or dom == 0):
        return True
    if target in PC_CTRL and reg in (0x0000, 0x0008, 0x0014, 0x0010):
        return True
    return False

def decode_fields(reg, value):
    if reg not in REGS:
        return "0x%x" % value
    _, name, fields = REGS[reg]
    parts = []
    for fn, lo, hi, ty in fields:
        if fn.startswith("RESERVED"):
            continue
        m = ((1 << (hi - lo + 1)) - 1) << lo
        fv = (value & m) >> lo
        if fv:
            parts.append("%s=%d(0x%x)" % (fn, fv, fv))
    return ", ".join(parts) if parts else "0"

def find_runs(data, minlen=6):
    runs, i, n = [], 0, len(data)
    while i + 8 <= n:
        reg, value, target = struct.unpack_from("<HIH", data, i)
        if valid_cmd(reg, target):
            j, cnt = i, 0
            cmds = []
            while j + 8 <= n:
                rg, vl, tg = struct.unpack_from("<HIH", data, j)
                if not valid_cmd(rg, tg):
                    break
                cmds.append((rg, vl, tg)); j += 8; cnt += 1
            if cnt >= minlen:
                runs.append((i, cmds))
                i = j
                continue
        i += 1
    return runs

def domain_of(reg, target):
    dom = target & 0xfffe
    for nm, v in TARGETS.items():
        if v == dom:
            return nm
    if target in PC_CTRL:
        return "PC"
    return "?0x%x" % dom

def main():
    for path in sys.argv[1:]:
        data = open(path, "rb").read()
        print("\n" + "=" * 78)
        print("%s  (%d bytes)" % (os.path.basename(path), len(data)))
        print("=" * 78)
        runs = find_runs(data)
        # keep runs that actually touch the PPU page
        ppu_runs = [r for r in runs if any(0x6000 <= rg < 0x7100 for rg, _, _ in r[1])]
        chosen = ppu_runs if ppu_runs else runs
        for off, cmds in chosen:
            print("--- regcmd run @0x%x  (%d words) ---" % (off, len(cmds)))
            for rg, vl, tg in cmds:
                nm = REGS[rg][1] if rg in REGS else "?"
                print("  [%-9s] %-26s val=0x%-8x %s" %
                      (domain_of(rg, tg), "%s(0x%x)" % (nm, rg), vl, decode_fields(rg, vl)))
        # crack helper: surface RECIP + kernel + mode + PC enable across the file
        print("  -- key fields --")
        for off, cmds in chosen:
            for rg, vl, tg in cmds:
                if rg in (0x6038, 0x603C, 0x6034, 0x6024, 0x6084, 0x7030, 0x6018, 0x601c, 0x6020):
                    print("    %-26s = 0x%x  (%s)" %
                          (REGS[rg][1] if rg in REGS else hex(rg), vl, decode_fields(rg, vl)))
                if tg == 0x81 and rg == 0x0008:
                    print("    PC_OPERATION_ENABLE         = 0x%x  (%s)" % (vl, decode_fields(rg, vl)))

if __name__ == "__main__":
    main()
