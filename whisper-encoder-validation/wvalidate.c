// Phase 1.2 NPU harness: run rocket_encoder_block_fp16 on REAL whisper weights
// exported by wprep.py, in two modes:
//   isolated: block L fed the golden's input gin[L]  (localizes per-block divergence)
//   chained : block L fed the previous NPU output     (real end-to-end fp16 fidelity)
// Writes rout_iso{L}.bin / rout_chn{L}.bin (fp16) for wcmp.py to score vs the f64 golden.
//
// build (on RK1):
//   gcc -O2 wvalidate.c -I <rocket-userspace>/include \
//       -I/usr/include/libdrm -L <rocket-userspace>/build \
//       -lrocketnpu -lpthread -lm -ldrm -o wvalidate
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rocket_npu.h"
#include "rocket_encoder.h"
static double nowms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec*1e-6; }

typedef _Float16 f16;
static char DIR[512];

static void *xread(const char *name, size_t bytes) {
    char p[640]; snprintf(p, sizeof p, "%s/%s", DIR, name);
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open %s: missing\n", p); exit(1); }
    void *buf = malloc(bytes);
    if (fread(buf, 1, bytes, f) != bytes) { fprintf(stderr, "short read %s\n", p); exit(1); }
    fclose(f); return buf;
}
static f16 *read_f16(const char *name, size_t n) { return (f16 *)xread(name, n * sizeof(f16)); }
static f16 *read_f32_to_f16(const char *name, size_t n) {
    float *t = (float *)xread(name, n * sizeof(float));
    f16 *o = malloc(n * sizeof(f16));
    for (size_t i = 0; i < n; i++) o[i] = (f16)t[i];
    free(t); return o;
}
static void write_f16(const char *name, const f16 *a, size_t n) {
    char p[640]; snprintf(p, sizeof p, "%s/%s", DIR, name);
    FILE *f = fopen(p, "wb"); fwrite(a, sizeof(f16), n, f); fclose(f);
}

int main(int argc, char **argv) {
    snprintf(DIR, sizeof DIR, "%s", argc > 1 ? argv[1] : "/tmp/wenc");
    char mpath[640]; snprintf(mpath, sizeof mpath, "%s/manifest.txt", DIR);
    FILE *mf = fopen(mpath, "r"); if (!mf) { perror("manifest"); return 1; }
    int T, d, nh, dff, nl; double eps;
    if (fscanf(mf, "%d %d %d %d %d %lf", &T, &d, &nh, &dff, &nl, &eps) != 6) return 1;
    fclose(mf);
    printf("manifest: T=%d d=%d nh=%d dff=%d nl=%d eps=%.2e\n", T, d, nh, dff, nl, eps);

    // per-block weights
    typedef struct { f16 *ln1g,*ln1b,*wq,*bq,*wk,*wv,*bv,*wo,*bo,*ln2g,*ln2b,*wf1,*bf1,*wf2,*bf2; } LW;
    LW *L = calloc(nl, sizeof(LW));
    char nm[64];
    for (int i = 0; i < nl; i++) {
        #define RD(field, short, cnt) do{ snprintf(nm,sizeof nm,"w%d_%s.bin",i,short); L[i].field = read_f16(nm,(cnt)); }while(0)
        RD(ln1g,"ln1g",d); RD(ln1b,"ln1b",d);
        RD(wq,"wq",(size_t)d*d); RD(bq,"bq",d);
        RD(wk,"wk",(size_t)d*d); RD(wv,"wv",(size_t)d*d); RD(bv,"bv",d);
        RD(wo,"wo",(size_t)d*d); RD(bo,"bo",d);
        RD(ln2g,"ln2g",d); RD(ln2b,"ln2b",d);
        RD(wf1,"wf1",(size_t)dff*d); RD(bf1,"bf1",dff);
        RD(wf2,"wf2",(size_t)d*dff); RD(bf2,"bf2",d);
        #undef RD
    }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed (%d): need /dev/accel\n", fd); return 2; }
    printf("rocket_open fd=%d\n", fd);

    size_t Td = (size_t)T * d;
    f16 *out = malloc(Td * sizeof(f16));

    // ---- isolated: each block fed the golden input gin[L] ----
    int chained_only = getenv("WVAL_CHAINED_ONLY") ? 1 : 0;
    for (int i = 0; !chained_only && i < nl; i++) {
        snprintf(nm, sizeof nm, "gin%d.bin", i);
        f16 *in = read_f32_to_f16(nm, Td);
        int rc = rocket_encoder_block_fp16(fd, T, d, nh, dff, in,
            L[i].ln1g, L[i].ln1b, L[i].wq, L[i].bq, L[i].wk, NULL /*bk*/, L[i].wv, L[i].bv,
            L[i].wo, L[i].bo, L[i].ln2g, L[i].ln2b, L[i].wf1, L[i].bf1, L[i].wf2, L[i].bf2,
            (float)eps, out);
        if (rc != 0) { fprintf(stderr, "block %d isolated rc=%d\n", i, rc); return 3; }
        snprintf(nm, sizeof nm, "rout_iso%d.bin", i); write_f16(nm, out, Td);
        free(in);
        printf("  isolated block %d done\n", i);
    }

    // ---- chained: block L fed previous NPU output, starting from gin0 ----
    f16 *x = read_f32_to_f16("gin0.bin", Td);
    double t_chain = nowms();
    for (int i = 0; i < nl; i++) {
        int rc = rocket_encoder_block_fp16(fd, T, d, nh, dff, x,
            L[i].ln1g, L[i].ln1b, L[i].wq, L[i].bq, L[i].wk, NULL, L[i].wv, L[i].bv,
            L[i].wo, L[i].bo, L[i].ln2g, L[i].ln2b, L[i].wf1, L[i].bf1, L[i].wf2, L[i].bf2,
            (float)eps, out);
        if (rc != 0) { fprintf(stderr, "block %d chained rc=%d\n", i, rc); return 3; }
        snprintf(nm, sizeof nm, "rout_chn%d.bin", i); write_f16(nm, out, Td);
        memcpy(x, out, Td * sizeof(f16));
    }
    fprintf(stderr, "[wval] chained %d blocks total=%.0f ms\n", nl, nowms() - t_chain);
    rocket_close(fd);
    printf("DONE\n");
    return 0;
}
