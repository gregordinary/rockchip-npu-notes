#!/usr/bin/env python3
# Idempotent: insert an env-gated (WHISPER_DUMP_ENC) dump of embd_conv + embd_enc
# right after the encoder graph computes in whisper_encode_internal. Temporary
# validation instrumentation (not a shipped change); backs up the original once.
import sys, os
SRC = sys.argv[1] if len(sys.argv) > 1 else "whisper.cpp/src/whisper.cpp"
s = open(SRC).read()
if "WHISPER_DUMP_ENC" in s:
    print("already patched"); sys.exit(0)
if not os.path.exists(SRC + ".bak"):
    open(SRC + ".bak", "w").write(s)
anchor = "    // cross\n    {\n        auto & sched = wstate.sched_cross.sched;"
assert s.count(anchor) == 1, f"anchor count = {s.count(anchor)} (expected 1)"
dump = r'''    if (getenv("WHISPER_DUMP_ENC")) {
        struct ggml_tensor *dts[2] = { wstate.embd_conv, wstate.embd_enc };
        const char *dnm[2] = { "embd_conv", "embd_enc" };
        for (int di = 0; di < 2; di++) {
            struct ggml_tensor *tt = dts[di];
            if (!tt) continue;
            size_t nb = ggml_nbytes(tt);
            void *buf = malloc(nb);
            ggml_backend_tensor_get(tt, buf, 0, nb);
            char pp[256]; snprintf(pp, sizeof pp, "/tmp/whisper_dump/%s.bin", dnm[di]);
            FILE *ff = fopen(pp, "wb"); if (ff) { fwrite(buf, 1, nb, ff); fclose(ff); }
            fprintf(stderr, "WHISPER_DUMP %s ne=[%lld,%lld,%lld,%lld] type=%d nbytes=%zu\n",
                    dnm[di], (long long)tt->ne[0], (long long)tt->ne[1], (long long)tt->ne[2],
                    (long long)tt->ne[3], (int)tt->type, nb);
            free(buf);
        }
    }

'''
s = s.replace(anchor, dump + anchor, 1)
open(SRC, "w").write(s)
print("patched OK")
