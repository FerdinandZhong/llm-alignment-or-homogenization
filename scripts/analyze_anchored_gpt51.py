import json, csv, math
from collections import defaultdict

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def extract_ba_user(rows):
    out = {}
    for record in rows:
        for uid, cats in record.items():
            per_user = {}
            for cat_answers in cats.values():
                for q_dict in cat_answers:
                    for q, v in q_dict.items():
                        per_user[q] = v.get("option_id") if isinstance(v, dict) else v
            out[uid] = per_user
    return out

def extract_anchored(rows):
    out = {}
    for record in rows:
        for uid, inner in record.items():
            per_user = {}
            results = inner.get("results", inner)
            for cat_answers in results.values():
                for q_dict in cat_answers:
                    for q, v in q_dict.items():
                        per_user[q] = v.get("option_id") if isinstance(v, dict) else v
            out[uid] = per_user
    return out

ba_user  = extract_ba_user(load_jsonl("wvs_values_results/gpt-5.1/BA_user_values_results/total_1000.jsonl"))
anchored = extract_anchored(load_jsonl("wvs_values_results/gpt-5.1/BA_anchored_values_results/total_1000.jsonl"))

gt = {}
with open("datasets/wvs_benchmarks/sampled_values_df.csv") as f:
    for row in csv.DictReader(f):
        gt[row["D_INTERVIEW"]] = row

with open("datasets/wvs_benchmarks/picked_questions.json") as f:
    pq = json.load(f)
all_q = {}
for cat_qs in pq.values():
    all_q.update(cat_qs)

print(f"BA_user={len(ba_user)}  Anchored={len(anchored)}  GT={len(gt)}  Questions={len(all_q)}")

# VAA
def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    if dx == 0 or dy == 0: return float('nan')
    return num / (dx * dy)

def compute_vaa(model_resp):
    rs = []
    for uid, preds in model_resp.items():
        if uid not in gt: continue
        pairs = []
        for q in all_q:
            p = preds.get(q); h = gt[uid].get(q)
            if p is not None and h not in (None, '', 'nan'):
                try: pairs.append((float(p), float(h)))
                except: pass
        if len(pairs) >= 5:
            r = pearson([x for x,_ in pairs], [y for _,y in pairs])
            if not math.isnan(r): rs.append(r)
    return rs

vu = compute_vaa(ba_user)
va = compute_vaa(anchored)
print(f"\nVAA BA_user:  n={len(vu)}  mean={sum(vu)/len(vu):.4f}")
print(f"VAA Anchored: n={len(va)}  mean={sum(va)/len(va):.4f}")
print(f"VAA Delta:    {sum(va)/len(va) - sum(vu)/len(vu):+.4f}")

# Ground truth as float
gt_f = {}
for uid, row in gt.items():
    d = {}
    for q in all_q:
        v = row.get(q)
        if v not in (None, '', 'nan'):
            try: d[q] = float(v)
            except: pass
    gt_f[uid] = d

# Demographics
with open("datasets/wvs_benchmarks/sampled_demographic_features.csv") as f:
    prof = {r["D_INTERVIEW"]: r for r in csv.DictReader(f)}

def l2(a, b):
    d = 0.0
    for q, meta in all_q.items():
        av = a.get(q); bv = b.get(q)
        if av is None or bv is None: continue
        try: av, bv = float(av), float(bv)
        except: continue
        if not (math.isfinite(av) and math.isfinite(bv)): continue
        sc = int(meta["answer_scale_max"]) - int(meta["answer_scale_min"])
        if sc > 0: d += ((av - bv) / sc) ** 2
    return math.sqrt(d)

def mean_vec(vecs):
    c = {}
    for q in all_q:
        vals = []
        for v in vecs:
            raw = v.get(q)
            if raw not in (None, '', 'nan'):
                try: vals.append(float(raw))
                except: pass
        if vals: c[q] = sum(vals) / len(vals)
    return c

def homog_rate(model_resp, attr):
    uid_g = {}
    for uid in set(model_resp) & set(gt_f):
        if uid in prof:
            g = prof[uid].get(attr, '')
            if g: uid_g[uid] = g
    groups = defaultdict(list)
    for uid, g in uid_g.items(): groups[g].append(uid)
    centroids = {g: mean_vec([gt_f[uid] for uid in uids if uid in gt_f]) for g, uids in groups.items()}
    toward = total = 0
    for uid, g in uid_g.items():
        if g not in centroids or uid not in gt_f or uid not in model_resp: continue
        dh = l2(gt_f[uid], centroids[g])
        dm = l2(model_resp[uid], centroids[g])
        toward += dm < dh
        total += 1
    return round(toward/total, 4) if total else 0.0, total

ATTRS = [
    "age", "continent_of_residence", "immigration_status",
    "highest_level_of_education", "socioeconomic_status", "occupation_group"
]

print(f"\n{'Attribute':<35} {'BA_user':>8} {'Anchored':>8} {'Delta':>7} {'n':>6}")
print("-" * 70)
totals_u = []; totals_a = []
for a in ATTRS:
    hu, nu = homog_rate(ba_user, a)
    ha, na = homog_rate(anchored, a)
    print(f"{a:<35} {hu:>8.4f} {ha:>8.4f} {ha-hu:>+7.4f} {nu:>6}")
    totals_u.append(hu); totals_a.append(ha)

print(f"\n{'MEAN':<35} {sum(totals_u)/len(totals_u):>8.4f} {sum(totals_a)/len(totals_a):>8.4f} {sum(totals_a)/len(totals_a)-sum(totals_u)/len(totals_u):>+7.4f}")

# Question coverage
sample_uid = list(anchored.keys())[0]
print(f"\nAnchored q-count (sample user): {len(anchored[sample_uid])}")
sample_uid2 = list(ba_user.keys())[0]
print(f"BA_user  q-count (sample user): {len(ba_user[sample_uid2])}")
