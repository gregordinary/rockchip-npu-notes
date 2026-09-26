#!/usr/bin/env python3
# Re-fit of the pinning-by-knob interaction model over the stored 2x2 campaigns, with no board time.
#
# The model [tuning-matrix.md, "The interaction is forced by an additive-time model"] is
#     I = (1 - a)(1 - b) / (1 - a - b + b*phi)
# with a = 1 - 1/K (K the knob's unpinned paired ratio), b = 1 - 1/P (P the base pin gain), and phi
# the fraction of host CORE-SECONDS the knob removes, read from busy_tot in the two PINNED arms.
# Three readings of phi are scored side by side:
#   A  the published one: phi as the core-seconds fraction (a residency cell's phi is overridden by
#      its rep-count-regression value, because the raw pinned ratio carries the ingest);
#   B  phi_wall = a(1 - 1/g)/b, the fraction of host WALL, closing the derivation's H through the
#      pin gain with g = busy_tot(base unpinned)/busy_tot(base pinned);
#   C  I == 1, what the substitution phi_wall = a collapses to.
# Every input is a <!--RO--> or <!--DATA--> row already on disk; per pass, paired within the pass.
# Result on 2026-09-03: A within 2.1 se on all six cells, B worse on five (3-7 se on four), C
# rejected at 4-10 se on the three largest knobs. The core-seconds phi is the quantity that fits.
import re, statistics as st, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ro-session') + '/'
def load(fn):
    rows={}
    for line in open(D+fn,errors='replace'):
        m=re.match(r'<!--RO (\d+)\t(\S+)\t(.*)-->',line)
        if m:
            p,arm,kv=int(m.group(1)),m.group(2),m.group(3)
            d=dict(x.split('=',1) for x in kv.split('\t') if '=' in x)
            rows.setdefault((p,arm),{}).update(busy=float(d['busy_tot']), wall=float(d.get('wall_s','nan')))
        m=re.match(r'<!--DATA (\d+)\t(\S+)\tpp2048\t([\d.]+)',line)
        if m: rows.setdefault((int(m.group(1)),m.group(2)),{})['ts']=float(m.group(3))
    return rows
cells=[ # file, base_unpin, base_pin, knob_unpin, knob_pin, phi_override(None=from pinned pair)
 ('9B x -ub 2048','trackd20-pin-ub-2x2-6pass.md','stock_unpin','stock_pin','ub_unpin','ub_pin',None),
 ('9B x quant residency','trackd16-pin9b-2x2-6pass.md','stock_unpin','stock_pin','qres_unpin','qres_pin',0.6645),
 ('12B F16 x MM_ASYM (23)','trackd23-asym-pin-12b-3pass.md','asym0_unpin','asym0_pin','stock_unpin','stock_pin',None),
 ('12B F16 x MM_ASYM (23b)','trackd23b-asym-pin-12b-repeat.md','asym0_unpin','asym0_pin','stock_unpin','stock_pin',None),
 ('12B F16 x f16 residency','trackd-res12b-2x2-3pass.md','stream_unpin','stream_pin','res_unpin','res_pin',0.227),
 ('12B Q4 x -ub 2048','trackd28-pin-ub-12bq4.md','stock_unpin','stock_pin','ub_unpin','ub_pin',None),
]
print(f"{'cell':26s} {'K':>6s} {'P':>6s} {'g':>6s} {'phi':>6s} {'phi_w':>6s} {'H/t':>5s} | {'I_meas':>7s} {'se':>6s} | {'A':>7s} {'resA':>5s} | {'B':>7s} {'resB':>5s} | {'C=1':>5s}")
for name,fn,bu,bp,ku,kp,phio in cells:
    r=load(fn); passes=sorted({p for p,_ in r})
    I=[];K=[];P=[];g=[];phi=[]
    for p in passes:
        try:
            t={a:r[(p,a)]['ts'] for a in (bu,bp,ku,kp)}; b={a:r[(p,a)]['busy'] for a in (bu,bp,ku,kp)}
        except KeyError: continue
        K.append(t[ku]/t[bu]); P.append(t[bp]/t[bu]); I.append((t[kp]/t[bp])/(t[ku]/t[bu]))
        g.append(b[bu]/b[bp]); phi.append(1-b[kp]/b[bp])
    n=len(I); Im=st.mean(I); se=st.stdev(I)/n**0.5
    a=1-1/st.mean(K); bb=1-1/st.mean(P); gg=st.mean(g); ph=phio if phio else st.mean(phi)
    A=(1-a)*(1-bb)/(1-a-bb+bb*ph)
    Ht=bb/(1-1/gg); phw=a/Ht
    B=(1-a)*(1-bb)/(1-a-bb+bb*phw)
    print(f"{name:26s} {st.mean(K):6.3f} {st.mean(P):6.3f} {gg:6.3f} {ph:6.3f} {phw:6.3f} {Ht:5.3f} | {Im:7.4f} {se:6.4f} | {A:7.4f} {(Im-A)/se:+5.1f} | {B:7.4f} {(Im-B)/se:+5.1f} | {(Im-1)/se:+5.1f}")

# --- Second table: for the cells whose knob arms carry no ingest, the removed work's OWN pinning
# speedup g_r = (C_base_unpin - C_knob_unpin)/(C_base_pin - C_knob_pin), and a variant model that
# weights phi by the removed work's share of the PIN GAIN, phi' = phi*(1-1/g_r)/(1-1/g). Refuted:
# the 9B micro-batch cell moves from +0.1 to -3.3 se and the 12B Q4_K_M cell has g_r = g.
print()
cells2=[('9B x -ub 2048','trackd20-pin-ub-2x2-6pass.md','stock_unpin','stock_pin','ub_unpin','ub_pin'),
       ('12B Q4 x -ub 2048','trackd28-pin-ub-12bq4.md','stock_unpin','stock_pin','ub_unpin','ub_pin'),
       ('12B F16 x MM_ASYM 23','trackd23-asym-pin-12b-3pass.md','asym0_unpin','asym0_pin','stock_unpin','stock_pin'),
       ('12B F16 x MM_ASYM 23b','trackd23b-asym-pin-12b-repeat.md','asym0_unpin','asym0_pin','stock_unpin','stock_pin')]
for name,fn,bu,bp,ku,kp in cells2:
    r=load(fn); ps=sorted({p for p,_ in r}); I=[];K=[];P=[];g=[];gr=[];phi=[]
    for p in ps:
        b={a:r[(p,a)]['busy'] for a in (bu,bp,ku,kp)}; t={a:r[(p,a)]['ts'] for a in (bu,bp,ku,kp)}
        K.append(t[ku]/t[bu]); P.append(t[bp]/t[bu]); I.append((t[kp]/t[bp])/(t[ku]/t[bu]))
        g.append(b[bu]/b[bp]); phi.append(1-b[kp]/b[bp]); gr.append((b[bu]-b[ku])/(b[bp]-b[kp]))
    a=1-1/st.mean(K); bb=1-1/st.mean(P); ph=st.mean(phi); G=st.mean(g); Gr=st.mean(gr); Im=st.mean(I); se=st.stdev(I)/len(I)**0.5
    A=(1-a)*(1-bb)/(1-a-bb+bb*ph); ph2=ph*(1-1/Gr)/(1-1/G); V=(1-a)*(1-bb)/(1-a-bb+bb*ph2)
    print(f"{name:24s} g={G:.3f} g_removed={Gr:.3f} phi={ph:.3f} phi'={ph2:.3f} | I {Im:.4f}+-{se:.4f} | model {A:.4f} ({(Im-A)/se:+.1f} se) | variant {V:.4f} ({(Im-V)/se:+.1f} se)")
