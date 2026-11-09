#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Read the DPU LUT program out of a compiled .rknn: every register write, named from
# Mesa's registers.xml, and the LUT table the program uploads.
#
# A register command is a little-endian 64-bit word: target in bits 63:48, value in
# 47:16, register offset in 15:0. The .rknn stores each task's program as an 8-byte-
# aligned run of such words, so a program is a run whose every top halfword is a block
# target. LUT_ACCESS_DATA (0x4104) writes are the table: one stray write in the register
# dump, then 513 LE entries after LUT_ACCESS_CFG 0x00020000 and 513 LO entries after
# 0x00030000.
#
# The field map defaults to ../ppu-rknn-capture/registers.xml (Mesa's, MIT); set
# ROCKET_REGISTERS_XML to use another copy.
#
# Usage: decode_lut_rknn.py [--table] file.rknn [...]
import itertools
import os
import struct
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.environ.get("ROCKET_REGISTERS_XML",
                     os.path.join(HERE, "..", "ppu-rknn-capture", "registers.xml"))
TARGETS = {0x0101, 0x0201, 0x0801, 0x1001, 0x2001, 0x4001, 0x8001, 0x0041, 0x0081}
LUT_DATA, LUT_ACCESS_CFG = 0x4104, 0x4100


def reg_names():
    names = {}
    for el in ET.parse(XML).getroot().iter():
        if el.tag.endswith("reg32"):
            names[int(el.get("offset"), 0)] = el.get("name")
    return names


def programs(path, min_len=12):
    blob = open(path, "rb").read()
    blob = blob[:len(blob) - len(blob) % 8]
    words = (w for (w,) in struct.iter_unpack("<Q", blob))
    for is_cmd, run in itertools.groupby(words, key=lambda w: (w >> 48) in TARGETS):
        prog = [(w >> 48, w & 0xFFFF, (w >> 16) & 0xFFFFFFFF) for w in run] if is_cmd else []
        if len(prog) >= min_len:
            yield prog


def lut_tables(prog):
    """The LE and LO tables, as signed 16-bit entries, in upload order."""
    tables, cur = {}, None
    for _, reg, val in prog:
        if reg == LUT_ACCESS_CFG:
            cur = {0x00020000: "LE", 0x00030000: "LO"}.get(val)
        elif reg == LUT_DATA and cur:
            v = val & 0xFFFF
            tables.setdefault(cur, []).append(v - 0x10000 if v & 0x8000 else v)
    return tables


def main(argv):
    show_table = "--table" in argv
    files = [a for a in argv if a != "--table"]
    if not files:
        sys.exit(__doc__ if __doc__ else "usage: decode_lut_rknn.py [--table] file.rknn")
    names = reg_names()
    for path in files:
        print("=" * 8, os.path.basename(path))
        for i, prog in enumerate(programs(path)):
            nlut = sum(1 for _, r, _ in prog if r == LUT_DATA)
            print(f"program {i}: {len(prog)} words, {nlut} LUT_ACCESS_DATA writes")
            for tgt, reg, val in prog:
                if reg != LUT_DATA:
                    print(f"  {tgt:04x} {reg:04x} {names.get(reg, '?'):28s} {val:08x}")
            for name, t in sorted(lut_tables(prog).items()):
                print(f"  {name}: {len(t)} entries, first {t[:3]}, last {t[-3:]}")
                if show_table:
                    print("   ", " ".join(str(v) for v in t))


if __name__ == "__main__":
    main(sys.argv[1:])
