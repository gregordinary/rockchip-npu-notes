# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rockchip-npu-notes authors
# CPU side: the NPU arm split vs the CPU decoder at 4/2/1 threads and with passive OpenMP waits.
import json, os, re, subprocess, time
D = "/path/to/workdir"
PIN = "/path/to/pin-governor.sh"
PROBE = D + "/rocket-userspace/build/whisper_decode_step_probe"
W = "/path/to/data/rebase/whisper.cpp-v1.9.4"
WDEC = D + "/cpu/wdec_cpu"
WAV = W + "/samples/jfk.wav"
MODELS = {"base": "/path/to/data/whisper/models/ggml-base.bin", "small": "/path/to/data/asr-svc/models/ggml-small.bin"}
OUT = open(D + "/campaign-cpu.jsonl", "a")
def pin(mode):
    r = subprocess.run(["bash", PIN, mode], capture_output=True, text=True); return (r.stdout + r.stderr).strip()
def env0(**kw):
    e = dict(os.environ); e.pop("SUDOPW", None); e.pop("GGML_BACKEND_PATH", None); e.update(kw); return e
def arm(model, kind):
    if kind == "npu":
        p = subprocess.run(["taskset", "-c", "4-7", PROBE, "step", model, "64"], env=env0(ROCKET_KACC="1"),
                           capture_output=True, text=True)
        m = re.search(r"RESULT step .*", p.stdout); rec = dict(model=model, kind=kind, rc=p.returncode)
        if m: rec.update({k: float(v) for k, v in re.findall(r"(\w+) ([0-9.]+)", m.group(0))})
        return [rec]
    e = env0(WDEC_BACKEND_DIR=W + "/build/bin")
    if kind == "cpu_passive": e["OMP_WAIT_POLICY"] = "passive"
    ths = "4,2,1" if kind == "cpu" else "4,2"
    p = subprocess.run(["taskset", "-c", "4-7", WDEC, MODELS[model], WAV, ths, "3"], env=e, capture_output=True, text=True)
    recs = []
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            r = dict(model=model, kind=kind, rc=p.returncode)
            r.update({k: float(v) for k, v in re.findall(r"(\w+) ([0-9.]+)", line)}); recs.append(r)
    if not recs: recs = [dict(model=model, kind=kind, rc=p.returncode, err=p.stderr[-300:])]
    return recs
try:
    print(pin("perfidle")); print(pin("show"), flush=True)
    kinds = ["npu", "cpu", "cpu_passive"]
    for model in MODELS:
        for kind in kinds: arm(model, kind)          # discarded warm-up
    for p in range(3):
        temps = pin("show").splitlines()[-1]
        for model in MODELS:
            for kind in kinds[p % 3:] + kinds[:p % 3]:
                pin("reset")
                for r in arm(model, kind):
                    r["pass"] = p; r["temps"] = temps; OUT.write(json.dumps(r) + "\n"); OUT.flush()
                    if kind == "npu":
                        print("PASS %d %-5s npu step %.2f mm %.2f host %.2f sattn %.2f xattn %.2f cpu %.2f sys %.2f" % (
                            p, model, r["step_med"], r["mm_med"], r["host_med"], r["sattn_med"], r["xattn_med"],
                            r["cpu_med"], r["sys_med"]), flush=True)
                    elif "threads" in r:
                        print("PASS %d %-5s %-11s rep %d t %d call %.2f step %.2f cpu %.2f" % (
                            p, model, kind, r["rep"], r["threads"], r["call_med"], r["step_med"], r["cpu_per_step"]), flush=True)
                    else:
                        print("PASS %d %s %s FAILED %s" % (p, model, kind, r), flush=True)
finally:
    print("restore:", pin("restore")); print(pin("show"), flush=True)
