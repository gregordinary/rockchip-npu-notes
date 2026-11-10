# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rockchip-npu-notes authors
# Pinned re-run of the decoder-step A/B: NPU probe vs whisper.cpp CPU, three rotated passes.
# SUDOPW comes from the environment (read from stdin by the caller); nothing is written to disk.
import json, os, re, resource, subprocess, sys, time
D = "/path/to/workdir"
PIN = "/path/to/pin-governor.sh"
PROBE = D + "/rocket-userspace/build/whisper_decode_step_probe"
W = "/path/to/data/rebase/whisper.cpp-v1.9.4"
CLI = W + "/build/bin/whisper-cli"
CLIP = {"full": W + "/samples/jfk.wav", "1s": D + "/cpu/jfk_1s.wav"}
MODELS = {"tiny": "/path/to/data/whisper/models/ggml-tiny.bin",
          "base": "/path/to/data/whisper/models/ggml-base.bin",
          "small": "/path/to/data/asr-svc/models/ggml-small.bin"}
OUT = open(D + "/campaign-wall.jsonl", "a")
def pin(mode):
    r = subprocess.run(["bash", PIN, mode], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()
def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime
def arm(model, kind):
    env = dict(os.environ); env.pop("GGML_BACKEND_PATH", None); env.pop("SUDOPW", None)
    if kind == "npu":
        env["ROCKET_KACC"] = "1"
        cmd = ["taskset", "-c", "4-7", PROBE, "step", model, "64"]
    else:
        cmd = ["taskset", "-c", "4-7", CLI, "-m", MODELS[model], "-f", CLIP[kind[4:]], "-t", "4", "-bs", "1",
               "-nf", "-l", "en", "-nt"]
    c0 = child_cpu(); t0 = time.monotonic()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    rec = dict(model=model, kind=kind, rc=p.returncode, wall=time.monotonic() - t0, cpu=child_cpu() - c0)
    if kind == "npu":
        m = re.search(r"RESULT step .*", p.stdout)
        if m:
            for k, v in re.findall(r"(\w+) ([0-9.]+)", m.group(0)):
                rec[k] = float(v)
            rec["agree"] = re.search(r"argmax_agree (\S+)", m.group(0)).group(1)
    else:
        m = re.search(r"decode time =\s*([0-9.]+) ms /\s*(\d+) runs \(\s*([0-9.]+) ms per run\)", p.stderr)
        if m: rec.update(dec_ms=float(m.group(1)), runs=int(m.group(2)), per=float(m.group(3)))
        rec["text"] = p.stdout.strip()[:60]
    return rec
try:
    print(pin("perfidle")); print(pin("show"), flush=True)
    for model in MODELS:                       # one discarded warm-up per arm
        for kind in ("npu", "cpu_full", "cpu_1s"): arm(model, kind)
    kinds = ["npu", "cpu_full", "cpu_1s"]
    for p in range(3):
        temps = pin("show").splitlines()[-1]
        for model in MODELS:
            order = kinds[p % 3:] + kinds[:p % 3]
            res = {}
            for kind in order:
                pin("reset")
                r = arm(model, kind); r["pass"] = p; r["temps"] = temps
                res[kind] = r; OUT.write(json.dumps(r) + "\n"); OUT.flush()
            n, f, s = res["npu"], res["cpu_full"], res["cpu_1s"]
            dcpu = 1e3 * (f["cpu"] - s["cpu"]) / (f["runs"] - s["runs"])
            print("PASS %d %-5s npu step %.2f mm %.2f host %.2f cpu %.2f sys %.2f agree %s | whisper.cpp step %.2f "
                  "decode_cpu %.2f | wall_ratio %.3f cpu_ratio %.3f | %s" % (
                  p, model, n["step_med"], n["mm_med"], n["host_med"], n["cpu_med"], n["sys_med"], n["agree"],
                  f["per"], dcpu, n["step_med"] / f["per"], n["cpu_med"] / dcpu, temps), flush=True)
finally:
    print("restore:", pin("restore")); print(pin("show"), flush=True)
