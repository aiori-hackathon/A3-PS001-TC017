# =====================================================================
# RANSCOPE-M  -  UNIFIED-INPUT PIPELINE (Google Colab)
#
# DESIGN
#   * ONE dataset (one CSV, all telemetry columns together) goes into the system.
#   * Each of the 4 pretrained layer models picks ONLY its own columns and ignores the rest
#     (enforced in code by layer_view(): a layer literally never sees other layers' columns).
#   * The 4 models run simultaneously (one thread each) and each returns a score in [0, 1]
#     (towards 0 = benign, towards 1 = malicious).
#   * ENSEMBLE: the 4 scores are weighted-averaged with the layer weights -> final score ->
#     verdict -> accuracy (when the CSV has a 'label' column).
#
#   Training (once): the 4 layer models are trained from the 4 separate layer CSVs
#   (each layer on its own dataset type) and saved to ONE artifact file.
#
# QUICK START
#   first time : upload MPVG/SFIM/EAP/CAIT_dataset.csv to /content, run this file
#                (trains, saves, writes a held-out UNIFIED test CSV, tests it)
#   new data   : RUN_TRAINING = False ; TEST_FILE = "/content/your_unified.csv" ; run
#                or in a cell:  results = test_dataset("/content/your_unified.csv")
#   have four layer CSVs from a new run and want one file?
#                make_unified_csv({"MPVG": p1, "SFIM": p2, "EAP": p3, "CAIT": p4}, "/content/unified.csv")
# =====================================================================

# %% ------------------------------------------------------------ 1. SETUP & CONFIG
import os, time, math, random, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
warnings.filterwarnings("ignore")

RUN_TRAINING = False           # backend-safe copy: never train on import
RUN_INFERENCE = False          # backend-safe copy: never run notebook tests on import

DATA_DIR = "/content"
TRAIN_FILES = {"MPVG": "MPVG_dataset.csv", "SFIM": "SFIM_dataset.csv",
               "EAP": "EAP_dataset.csv", "CAIT": "CAIT_dataset.csv"}
TEST_DIR = "/content/test_data"
# >>> ONE unified CSV: the columns of all layers together. Point this at any new dataset. <<<
TEST_FILE = f"{TEST_DIR}/unified_test.csv"
ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ranscope_m_artifacts.pt")
FILES = list(TRAIN_FILES)

# ---- ensemble settings (editable at test time, no retraining needed) ----
ENSEMBLE_WEIGHTS = {"MPVG": 0.18, "SFIM": 0.26, "EAP": 0.31, "CAIT": 0.25}
THRESHOLD = 0.5            # fused probability >= THRESHOLD  ->  MALICIOUS
PERSISTENCE = 1            # consecutive windows above threshold before an ALERT

# ---- training settings ----
SEED, TEST_FRAC = 42, 0.20
EPOCHS, PATIENCE, LR, BATCH = 200, 25, 1e-3, 32
BALANCED_LOSS = True                   # class-balanced BCE
DROP_POST_ENCRYPTION_FEATURES = True   # hide enc_count / locked_count (already-encrypted evidence)
USE_EAP_EXTRA_RUNS = False             # EAP-only runs have identical entropy in both classes
TRAIN_KS = [1, 2, 3, 5, None]          # train on first-k-window prefixes (None = full run)
RUN_UNSEEN_TEST = False              # slow (~3x training): train WITHOUT one ransomware scenario, test on it

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

EXT = ["txt", "csv", "json", "pdf", "jpg", "png", "docx", "md", "log"]
if not DROP_POST_ENCRYPTION_FEATURES:
    EXT += ["enc", "locked"]
EXT_COLS = [f"{e}_count" for e in EXT]

RUNS = {}      # layer -> {run_id: DataFrame of that run's windows}  (read by the featurisers)

def load_layer_csv(path):
    df = pd.read_csv(path).sort_values(["run_id", "window_start_ms"]).reset_index(drop=True)
    return {rid: g.reset_index(drop=True) for rid, g in df.groupby("run_id")}, df

def first_k(df, k):
    """first k windows of a run (k=None -> whole run). This is what makes inference 'real time':
    at window t each layer only sees windows 1..t."""
    return df if k is None else df.iloc[:k]

class Spec:
    """One layer = featuriser (inputs) + model factory (new_model) + scaler state."""
    name = ""
    def fit_scaler(self, rids): pass
    def scaler_state(self): return {}
    def load_scaler(self, st): pass
    def inputs(self, rids, k=None): raise NotImplementedError
    def new_model(self): raise NotImplementedError

# %% ------------------------------------------------------------ 2a. LAYER 1 - MPVG
# Micro-Permission Volatility Graph
#   Telemetry : android.app.Activity.requestPermissions(), Context.checkSelfPermission()
#   Signal    : permission requests escalate in bursts during ransomware reconnaissance
#   Per window: P_t = checkSelfPermission_count + requestPermissions_count
#   Features (8, log1p then standardised):
#       1 volatility     Vp = sum_t |P_t - P_(t-1)|   (P_0 = 0)
#       2 escalation     permission events per second
#       3 burst          max P_t over windows
#       4 burst windows  number of windows with >= 2 permission events
#       5 total checkSelfPermission     6 total requestPermissions
#       7 mean P_t                      8 share of windows with any permission event
#   Network   : MLP  8 -> 64 -> 32 -> 1   (ReLU, dropout 0.2)   ~2.7k parameters
class MPVGNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)              # logit; sigmoid -> P(malicious)

class MPVGSpec(Spec):
    name = "MPVG"
    def _feat(self, df, k):
        d = first_k(df, k)
        chk = d["MPVG_checkSelfPermission_count"].values.astype(float)
        req = d["MPVG_requestPermissions_count"].values.astype(float)
        P = chk + req
        vol = np.abs(P - np.concatenate([[0.0], P[:-1]])).sum()
        esc = P.sum() / max(len(P) * 5.0, 1.0)
        return [vol, esc, P.max(), float((P >= 2).sum()), chk.sum(), req.sum(), P.mean(), float((P > 0).mean())]
    def _raw(self, rids, k):
        return np.log1p(np.array([self._feat(RUNS["MPVG"][r], k) for r in rids]))
    def fit_scaler(self, rids):
        X = self._raw(rids, None); self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
    def scaler_state(self): return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}
    def load_scaler(self, st): self.mu, self.sd = np.array(st["mu"]), np.array(st["sd"])
    def inputs(self, rids, k=None):
        return (torch.tensor((self._raw(rids, k) - self.mu) / self.sd, dtype=torch.float32),)
    def new_model(self): return MPVGNet(8)

# %% ------------------------------------------------------------ 2b. LAYER 2 - SFIM
# Semantic File-access Intent Modelling (graph neural network)
#   Telemetry : FileOutputStream / File.renameTo() / File.delete() + file-system events
#   Signal    : ransomware sweeps directories and rewrites / renames / deletes files
#   Graph (one per run, built from all windows seen so far):
#       node 0      = process
#       10 op nodes = read_open, write_open, delete, rename (hooks) +
#                     create, modify, close_write, delete, moved_from, moved_to (file events)
#       file-type nodes = txt, csv, json, pdf, jpg, png, docx, md, log
#       edges       = process <-> every node, weight = log(1 + count), self-loops added,
#                     symmetric normalisation  A_hat = D^-1/2 (A + I) D^-1/2
#   Node features (26): node-identity one-hot, log count, share of total, and for the
#       process node 4 extras: access locality, rename-to-delete ratio, entropy, depth
#   Network   : GCN(26->64) -> GCN(64->64) -> global mean pool -> 64 -> 32 -> 1  ~8k parameters
#               (pure PyTorch, dense adjacency, no PyTorch-Geometric needed)
OP_COLS = ["SFIM_file_read_open_count", "SFIM_file_write_open_count", "SFIM_file_delete_count",
           "SFIM_file_rename_count", "create_count", "modify_count", "close_write_count",
           "delete_count", "moved_from_count", "moved_to_count"]
G_COLS = OP_COLS + EXT_COLS
N_NODES = 1 + len(G_COLS)
G_FEAT = N_NODES + 2 + 4

class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.lin = nn.Linear(i, o)
    def forward(self, x, a):
        return F.relu(torch.bmm(a, self.lin(x)))        # H' = ReLU(A_hat X W)

class SFIMNet(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.g1, self.g2 = GCNLayer(f, 64), GCNLayer(64, 64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x, a):
        h = self.g2(F.dropout(self.g1(x, a), 0.1, self.training), a)
        return self.head(h.mean(1)).squeeze(-1)         # global mean pooling over nodes

class SFIMSpec(Spec):
    name = "SFIM"
    def _graph(self, df, k):
        d = first_k(df, k)
        cnt = d[G_COLS].sum().values.astype(float)
        ev = max(d["file_event_count"].sum(), 1.0)
        w = np.maximum(d["entropy_observation_count"].values, 1)
        ent = float((d["avg_entropy"].values * w).sum() / w.sum())
        wd = np.maximum(d["file_event_count"].values, 1)
        depth = float((d["avg_depth"].values * wd).sum() / wd.sum())
        locality = float(d["unique_path_count"].sum() / ev)          # access locality
        r2d = math.log1p(cnt[3] / (cnt[2] + 1.0))                    # rename-to-delete ratio
        X = np.zeros((N_NODES, G_FEAT), dtype=np.float32)
        X[np.arange(N_NODES), np.arange(N_NODES)] = 1.0              # node identity
        X[0, N_NODES] = math.log1p(ev) / 8.0; X[0, N_NODES + 1] = 1.0
        X[0, N_NODES + 2:] = [locality, r2d, ent / 8.0, depth / 4.0]
        lc = np.log1p(cnt)
        X[1:, N_NODES] = lc / 8.0
        X[1:, N_NODES + 1] = cnt / max(cnt.sum(), 1.0)
        A = np.eye(N_NODES, dtype=np.float32)
        ew = lc / (lc.max() + 1e-6)
        A[0, 1:] = ew; A[1:, 0] = ew
        dinv = 1.0 / np.sqrt(A.sum(1))
        return X, (A * dinv[:, None] * dinv[None, :]).astype(np.float32)
    def inputs(self, rids, k=None):
        g = [self._graph(RUNS["SFIM"][r], k) for r in rids]
        return (torch.tensor(np.stack([a for a, _ in g])), torch.tensor(np.stack([b for _, b in g])))
    def new_model(self): return SFIMNet(G_FEAT)

# %% ------------------------------------------------------------ 2c. LAYER 3 - EAP
# Entropy Acceleration Predictor (LSTM)
#   Telemetry : javax.crypto.Cipher.doFinal() + Shannon entropy of written data
#   Signal    : encryption makes entropy rise convexly -> track its 2nd derivative
#   Per window t (one sequence step), 6 standardised features:
#       H_t (avg entropy), max entropy, dH = H_t - H_(t-1),
#       d2H = dH_t - dH_(t-1)   <- entropy ACCELERATION,
#       log(1 + Cipher.doFinal count), log(1 + entropy observation count)
#   Network   : 2-layer LSTM (hidden 64, dropout 0.2) over the padded sequence
#               (packed, so padding is ignored) -> last hidden state -> 64 -> 32 -> 1  ~54k params
#   Note      : a 1-window run has d2H = 0; acceleration only exists in multi-window runs.
EAP_D, MAX_T = 6, 17

class EAPNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(d, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x, ln):
        pk = nn.utils.rnn.pack_padded_sequence(x, ln.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(pk)
        return self.head(h[-1]).squeeze(-1)

class EAPSpec(Spec):
    name = "EAP"
    def _seq(self, df, k):
        d = first_k(df, k)
        H = d["avg_entropy"].values.astype(float)
        dH = np.diff(H, prepend=H[0])
        d2H = np.diff(dH, prepend=dH[0])                              # second derivative
        return np.stack([H, d["max_entropy"].values, dH, d2H,
                         np.log1p(d["EAP_cipher_doFinal_count"].values),
                         np.log1p(d["entropy_observation_count"].values)], 1)
    def fit_scaler(self, rids):
        S = np.concatenate([self._seq(RUNS["EAP"][r], None) for r in rids])
        self.mu, self.sd = S.mean(0), S.std(0) + 1e-6
    def scaler_state(self): return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}
    def load_scaler(self, st): self.mu, self.sd = np.array(st["mu"]), np.array(st["sd"])
    def inputs(self, rids, k=None):
        seqs = [(self._seq(RUNS["EAP"][r], k) - self.mu) / self.sd for r in rids]
        X = np.zeros((len(seqs), MAX_T, EAP_D), dtype=np.float32)
        for i, s in enumerate(seqs): X[i, :len(s[:MAX_T])] = s[:MAX_T]
        ln = np.array([min(len(s), MAX_T) for s in seqs])
        return (torch.tensor(X), torch.tensor(ln, dtype=torch.long))
    def new_model(self): return EAPNet(EAP_D)

# %% ------------------------------------------------------------ 2d. LAYER 4 - CAIT
# Cognitive API Intent Transformer
#   Telemetry : API-call activity (file create/modify/close_write/delete/moved_from/
#               moved_to + camera_id_list / camera_open)
#   Tokens    : each non-zero API count in each window becomes ONE token
#               token = (API type, log2 count bucket 1..9); sequence capped at 128 tokens
#   Network   : learned 128-d embeddings (vocab 74) + sinusoidal positional encoding
#               -> 2 Transformer encoder layers (4 heads, feed-forward 256, dropout 0.1)
#               -> masked mean pooling -> linear -> intent logit   ~275k parameters
#   Output    : sigmoid(logit) = intent score in (0, 1)
CAIT_APIS = ["create_count", "modify_count", "close_write_count", "delete_count",
             "moved_from_count", "moved_to_count", "CAIT_camera_id_list_count", "CAIT_camera_open_count"]
N_BUCKET, MAX_LEN = 9, 128
NOOP = 1 + len(CAIT_APIS) * N_BUCKET
VOCAB = NOOP + 1

class CAITNet(nn.Module):
    def __init__(self, vocab, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        pe = torch.zeros(MAX_LEN, d)
        pos = torch.arange(MAX_LEN).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, 4, 256, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)
    def forward(self, tok, mask):                     # mask: True = real token
        h = self.enc(self.emb(tok) + self.pe[: tok.size(1)], src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

class CAITSpec(Spec):
    name = "CAIT"
    def _tokens(self, df, k):
        d = first_k(df, k); toks = []
        for _, row in d.iterrows():
            for j, c in enumerate(CAIT_APIS):
                v = row[c]
                if v > 0:
                    toks.append(1 + j * N_BUCKET + (min(int(np.log2(1 + v)) + 1, N_BUCKET) - 1))
        return (toks or [NOOP])[:MAX_LEN]
    def inputs(self, rids, k=None):
        seqs = [self._tokens(RUNS["CAIT"][r], k) for r in rids]
        T = np.zeros((len(seqs), MAX_LEN), dtype=np.int64)
        for i, s in enumerate(seqs): T[i, :len(s)] = s
        return (torch.tensor(T), torch.tensor(T > 0))
    def new_model(self): return CAITNet(VOCAB)


# %% ------------------------------------------------------------ 2e. WHICH COLUMNS EACH LAYER USES
# A unified CSV contains the shared file-activity columns plus the layer-specific columns
# (MPVG_*, SFIM_*, EAP_*, CAIT_*). Each layer selects only the columns below.
META = ["run_id", "window_start_ms", "window_end_ms"]
LAYER_COLUMNS = {
    "MPVG": ["MPVG_checkSelfPermission_count", "MPVG_requestPermissions_count"],
    "SFIM": OP_COLS + EXT_COLS + ["file_event_count", "unique_path_count", "avg_depth",
                                  "avg_entropy", "entropy_observation_count"],
    "EAP":  ["avg_entropy", "max_entropy", "entropy_observation_count", "EAP_cipher_doFinal_count"],
    "CAIT": CAIT_APIS,
}

def layer_view(df, layer):
    """the ONLY columns this layer is allowed to see (everything else is dropped)"""
    cols = list(dict.fromkeys(META + LAYER_COLUMNS[layer]))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{layer}: columns missing from the dataset: {missing}")
    return df[cols]

def build_runs(df, layer):
    v = layer_view(df, layer).sort_values(["run_id", "window_start_ms"])
    return {rid: g.reset_index(drop=True) for rid, g in v.groupby("run_id")}

def merge_layer_dfs(dfs):
    """merge the 4 per-layer CSVs (DataFrames) into ONE unified table on (run_id, window_start_ms).
    Shared columns are identical across layers; layer-specific columns are attached; a layer that
    has no row for a window gets 0 for its own columns."""
    key = ["run_id", "window_start_ms"]
    is_spec = lambda c: any(c.startswith(f"{m}_") for m in FILES)
    base = pd.concat([dfs[l][[c for c in dfs[l].columns if not is_spec(c)]] for l in FILES])
    out = base.drop_duplicates(key, keep="first")
    for l in FILES:
        sc = [c for c in dfs[l].columns if c.startswith(f"{l}_")]
        out = out.merge(dfs[l][key + sc].drop_duplicates(key), on=key, how="left")
        out[sc] = out[sc].fillna(0)
    return out.sort_values(key).reset_index(drop=True)

def make_unified_csv(files, out_path):
    """files = {"MPVG": csv, "SFIM": csv, "EAP": csv, "CAIT": csv}"""
    u = merge_layer_dfs({l: pd.read_csv(files[l]) for l in FILES})
    u.to_csv(out_path, index=False); print(f"Wrote unified CSV: {out_path}  ({len(u)} windows, {u.run_id.nunique()} runs)")
    return u

def print_layer_columns():
    print("\nCOLUMNS EACH LAYER USES (all other columns are ignored)")
    for l in FILES:
        print(f"  {l:5s} {len(set(LAYER_COLUMNS[l]))} columns: {sorted(set(LAYER_COLUMNS[l]))}")

SPECS = {"MPVG": MPVGSpec, "SFIM": SFIMSpec, "EAP": EAPSpec, "CAIT": CAITSpec}

def print_model_summary():
    print("\nMODEL SUMMARY")
    for l in FILES:
        m = SPECS[l]().new_model()
        n = sum(p.numel() for p in m.parameters())
        print(f"  {l:5s} {type(m).__name__:9s} parameters: {n:,}  (~{n*4/1e6:.2f} MB float32)")

# %% ------------------------------------------------------------ 3. ENSEMBLE LAYER
class WeightedEnsemble:
    """Final fusion: p_final = sum_l w_l * p_l   (weights normalised to sum to 1).
    Used per time window; a window is flagged when p_final >= threshold, and an ALERT is
    raised after `persistence` consecutive flagged windows. If a layer has no data yet,
    it is skipped and the remaining weights are renormalised."""
    def __init__(self, weights=None, threshold=None, persistence=None):
        self.w = dict(weights or ENSEMBLE_WEIGHTS)
        self.threshold = THRESHOLD if threshold is None else threshold
        self.persistence = PERSISTENCE if persistence is None else persistence
    def fuse(self, P, avail=None):
        num, den = 0.0, 0.0
        for l in FILES:
            m = 1.0 if avail is None else np.asarray(avail[l], dtype=float)
            num = num + self.w[l] * np.nan_to_num(np.asarray(P[l], dtype=float)) * m
            den = den + self.w[l] * m
        return num / np.maximum(den, 1e-9)
    def alerts(self, PF, valid):
        """PF, valid: arrays (windows, runs). Returns boolean alert matrix."""
        above = (PF >= self.threshold) & valid
        out = np.zeros_like(above)
        for t in range(PF.shape[0]):
            if t + 1 >= self.persistence:
                out[t] = above[t + 1 - self.persistence: t + 1].all(0)
        return out

ENSEMBLE = WeightedEnsemble()

def metrics(y, p, thr=THRESHOLD):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
                FPR=fp / max(fp + tn, 1), AUC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                TP=tp, FP=fp, TN=tn, FN=fn)


# %% ------------------------------------------------------------ 4. TRAIN ONCE + SAVE
def _take(inputs, idx, dev=DEVICE):
    return [t[idx].to(dev) for t in inputs]

@torch.no_grad()
def predict(model, inputs, bs=128):
    model.eval(); out = []; n = len(inputs[0])
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n))
        out.append(torch.sigmoid(model(*_take(inputs, idx))).cpu())
    return torch.cat(out).numpy()

def make_xy(spec, rids, ks, LABEL):
    """prefix-augmented set: first-k-window views of every run (only if the run is longer than k)."""
    layer = spec.name; Xs, ys = [], []
    for k in ks:
        sel = [r for r in rids if k is None or len(RUNS[layer][r]) > k]
        if not sel: continue
        Xs.append(spec.inputs(sel, k)); ys += [LABEL[r] for r in sel]
    X = tuple(torch.cat([x[i] for x in Xs]) for i in range(len(Xs[0])))
    return X, torch.tensor(ys, dtype=torch.float32)

def fit(spec, tr_rids, va_rids, ks, LABEL):
    (Xtr, ytr), (Xva, yva) = make_xy(spec, tr_rids, ks, LABEL), make_xy(spec, va_rids, ks, LABEL)
    model = spec.new_model().to(DEVICE)
    pw = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if BALANCED_LOSS else 1.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.binary_cross_entropy_with_logits(model(*_take(Xtr, idx)), ytr[idx].to(DEVICE), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(*_take(Xva, torch.arange(len(yva)))),
                                                    yva.to(DEVICE), pos_weight=pw).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k_: v.detach().clone().cpu() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    print(f"    {spec.name}: stopped at epoch {ep+1}, best validation loss {best:.4f}")
    return model


def train_and_save():
    """Train each layer on ITS OWN dataset (separate CSVs), save one artifact, and write a
    held-out test set as ONE unified CSV."""
    LABEL, SCEN, RAW = {}, {}, {}
    for layer, fname in TRAIN_FILES.items():
        RAW[layer] = pd.read_csv(os.path.join(DATA_DIR, fname))
        RUNS[layer] = build_runs(RAW[layer], layer)            # layer sees only its own columns
        g = RAW[layer].groupby("run_id")[["label", "scenario"]].first()
        for rid, row in g.iterrows():
            LABEL[rid] = int(row["label"] == "malicious"); SCEN.setdefault(rid, row["scenario"])
    common = sorted(set.intersection(*[set(RUNS[l]) for l in FILES]))
    extra = sorted(set(RUNS["EAP"]) - set(common)) if USE_EAP_EXTRA_RUNS else []
    tr_all, te = train_test_split(common, test_size=TEST_FRAC, random_state=SEED,
                                  stratify=[SCEN[r] for r in common])
    tr, va = train_test_split(tr_all, test_size=0.15, random_state=SEED, stratify=[SCEN[r] for r in tr_all])
    print(f"Runs: train={len(tr)}  validation={len(va)}  held-out test={len(te)}")

    os.makedirs(TEST_DIR, exist_ok=True)                       # held-out telemetry -> ONE unified CSV
    unified = merge_layer_dfs({l: RAW[l][RAW[l].run_id.isin(te)] for l in FILES})
    unified.to_csv(TEST_FILE, index=False)
    print(f"Wrote ONE unified held-out test CSV: {TEST_FILE}  ({len(unified)} windows, {unified.run_id.nunique()} runs)")

    art = {"layers": {}, "weights": ENSEMBLE_WEIGHTS, "threshold": THRESHOLD,
           "config": {"drop_post": DROP_POST_ENCRYPTION_FEATURES, "ext": EXT, "train_ks": list(TRAIN_KS),
                      "columns": {l: sorted(set(LAYER_COLUMNS[l])) for l in FILES}}}
    t0 = time.time()
    for layer in FILES:
        set_seed(SEED)
        spec = SPECS[layer](); tr_l = tr + (extra if layer == "EAP" else [])
        spec.fit_scaler(tr_l)
        model = fit(spec, tr_l, va, TRAIN_KS, LABEL)
        art["layers"][layer] = {"state": {k_: v.cpu() for k_, v in model.state_dict().items()},
                                "scaler": spec.scaler_state()}
    torch.save(art, ARTIFACT)
    print(f"Saved the 4 pretrained models -> {ARTIFACT}  ({(time.time()-t0)/60:.1f} min)")

def load_pretrained(path=ARTIFACT):
    art = torch.load(path, map_location="cpu")
    assert art["config"]["drop_post"] == DROP_POST_ENCRYPTION_FEATURES and art["config"]["ext"] == EXT, \
        "Feature config differs from training - set DROP_POST_ENCRYPTION_FEATURES to the training value."
    specs, models = {}, {}
    for l in FILES:
        sp = SPECS[l](); sp.load_scaler(art["layers"][l]["scaler"])
        m = sp.new_model(); m.load_state_dict(art["layers"][l]["state"]); m.to(DEVICE).eval()
        specs[l], models[l] = sp, m
    return specs, models


# %% ------------------------------------------------------------ 5. TEST ONE UNIFIED DATASET
from concurrent.futures import ThreadPoolExecutor

def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n"""
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))

def test_dataset(
    data,
    ensemble=None,
    save_to="/content/ranscope_m_results.xlsx",
    per_window=True,
    parallel=True,
    download=True,
    artifact=ARTIFACT,
    export_excel=True,
    pretrained_specs=None,
    pretrained_models=None,
):
    """data = path to ONE csv (or a DataFrame) that holds all columns.
    Each pretrained layer takes only its own columns, the 4 layers run simultaneously (threads),
    each returns a score in [0,1], and the ensemble weighted-averages the scores."""
    ens = ensemble or ENSEMBLE

    if pretrained_specs is not None and pretrained_models is not None:
        specs, models = pretrained_specs, pretrained_models
    else:
        specs, models = load_pretrained(artifact)

    df = (
        pd.read_csv(data)
        if isinstance(data, (str, os.PathLike))
        else data.copy()
    )
    df = df.sort_values(
        ["run_id", "window_start_ms"]
    ).reset_index(drop=True)

    active, skipped = [], {}
    for l in FILES:                                            # can this layer run on this dataset?
        miss = [c for c in dict.fromkeys(META + LAYER_COLUMNS[l]) if c not in df.columns]
        if miss: skipped[l] = miss
        else: active.append(l)
    if not active: raise ValueError("None of the four layers finds its columns in this dataset.")
    for l, miss in skipped.items():
        print(f"WARNING: layer {l} skipped (missing columns {miss}); remaining weights are renormalised.")
    avail = {l: float(l in active) for l in FILES}
    for l in active: RUNS[l] = build_runs(df, l)               # each layer's private view of the data

    rids = sorted(df.run_id.unique())
    T_of = df.groupby("run_id").size().reindex(rids).values
    Tmax = int(T_of.max())
    win_s = float((df["window_end_ms"] - df["window_start_ms"]).median() / 1000.0)
    print(f"\nDataset: {len(df)} windows | {len(rids)} runs | window {win_s:.0f}s | up to {Tmax} windows per run")
    print_layer_columns()
    print("Layers running:", active, "| weights:", ens.w, "| threshold", ens.threshold)

    def score_layers(k):                                       # 4 layers simultaneously
        def one(l): return l, predict(models[l], specs[l].inputs(rids, k))
        if parallel:
            with ThreadPoolExecutor(max_workers=len(active)) as ex: out = dict(ex.map(one, active))
        else:
            out = dict(map(one, active))
        for l in FILES: out.setdefault(l, np.zeros(len(rids)))
        return out

    P = score_layers(None)                                     # layer scores for the full run
    pf = ens.fuse(P, avail)                                    # ENSEMBLE: weighted average of the scores
    run_df = pd.DataFrame({"run_id": rids, "windows": T_of, **{f"p_{l}": P[l] for l in FILES}, "p_final": pf})
    run_df["verdict"] = np.where(pf >= ens.threshold, "MALICIOUS", "BENIGN")
    meta = df.groupby("run_id").first()
    if "scenario" in df.columns: run_df["scenario"] = run_df.run_id.map(meta["scenario"])

    out = {"run_df": run_df, "df": df, "rids": rids, "win_s": win_s, "metrics_df": None}
    if "label" in df.columns:
        y = run_df.run_id.map(meta["label"].eq("malicious").astype(int)).values; run_df["true_label"] = y
        res = {l: metrics(y, P[l], ens.threshold) for l in active}
        res["ENSEMBLE (weighted avg)"] = metrics(y, pf, ens.threshold)
        mdf = pd.DataFrame(res).T; out["metrics_df"] = mdf
        print("\n" + "=" * 78 + "\nRESULTS (end-of-run verdict)\n" + "=" * 78)
        print(mdf.round(4).to_string())
        e = res["ENSEMBLE (weighted avg)"]; n = len(y); npos = int(y.sum()); nneg = n - npos
        a_lo, a_hi = wilson(int(e["TP"] + e["TN"]), n)
        r_lo, r_hi = wilson(int(e["TP"]), npos); f_lo, f_hi = wilson(int(e["FP"]), nneg)
        bal = (e["Recall"] + (1 - e["FPR"])) / 2
        print(f"\nFINAL ENSEMBLE  accuracy {e['Accuracy']:.4f} (95% CI {a_lo:.3f}-{a_hi:.3f}) | "
              f"balanced accuracy {bal:.4f} | recall {e['Recall']:.4f} (CI {r_lo:.3f}-{r_hi:.3f}) | "
              f"FPR {e['FPR']:.4f} (CI {f_lo:.3f}-{f_hi:.3f})   [n={n}: {npos} malicious, {nneg} benign]")
        print(f"\nConfusion matrix\n            benign  malicious\n"
              f" benign     {int(e['TN']):6d}  {int(e['FP']):9d}\n malicious  {int(e['FN']):6d}  {int(e['TP']):9d}")

        diag = pd.DataFrame({l: {"mean_score": P[l].mean(), "std_score": P[l].std(),
                                 "distinct_values": len(set(np.round(P[l], 3))), "AUC": mdf.loc[l, "AUC"]}
                             for l in active}).T
        def _flag(r):
            if r.AUC < 0.65 or r.std_score < 0.05: return "WEAK / NO SIGNAL"
            if r.AUC >= 0.999: return "PERFECT - check for trivially separable data"
            return "ok"
        diag["status"] = diag.apply(_flag, axis=1)
        print("\nLAYER DIAGNOSTICS\n", diag.round(3).to_string())
        abl = {"all layers (weighted avg)": metrics(y, pf, ens.threshold)}
        for d_ in active:
            av = {l: (0.0 if (l == d_ or l not in active) else 1.0) for l in FILES}
            abl[f"without {d_}"] = metrics(y, ens.fuse(P, av), ens.threshold)
        abl_df = pd.DataFrame(abl).T[["Accuracy", "Recall", "FPR", "AUC"]]
        print("\nENSEMBLE ABLATION\n", abl_df.round(3).to_string())
        out["diag_df"], out["ablation_df"] = diag, abl_df
        if "scenario" in run_df:
            run_df["correct"] = ((pf >= ens.threshold).astype(int) == y).astype(int)
            print("\nPER-SCENARIO share correct\n", run_df.groupby("scenario").correct.agg(["size", "mean"]).round(3).to_string())
    else:
        print("\n(no 'label' column -> scores and verdicts only)\n", run_df.round(3).head(15).to_string(index=False))

    win_df = None
    if per_window:                                             # one fused score per time window
        rows = []
        for t in range(1, Tmax + 1):
            Pt = score_layers(t); ft = ens.fuse(Pt, avail)
            for j, r in enumerate(rids):
                if t <= T_of[j]:
                    rows.append({"run_id": r, "window": t, "time_s": t * win_s,
                                 **{f"p_{l}": Pt[l][j] for l in FILES}, "p_final": ft[j],
                                 "flag": int(ft[j] >= ens.threshold)})
        win_df = pd.DataFrame(rows); out["win_df"] = win_df

        if export_excel:
          with pd.ExcelWriter(save_to, engine="openpyxl") as xw:
            run_df.round(5).to_excel(xw, "Per_Run_Final", index=False)

            if win_df is not None:
                win_df.round(5).to_excel(xw, "Per_Window", index=False)

            if out["metrics_df"] is not None:
                out["metrics_df"].round(4).to_excel(xw, "Metrics")

            if "diag_df" in out:
                out["diag_df"].round(4).to_excel(xw, "Layer_Diagnostics")

            if "ablation_df" in out:
                out["ablation_df"].round(4).to_excel(xw, "Ensemble_Ablation")

            pd.DataFrame([
                {
                    "layer": layer,
                    "columns_used": ", ".join(
                        sorted(set(LAYER_COLUMNS[layer]))
                    ),
                }
                for layer in FILES
            ]).to_excel(xw, "Columns_Per_Layer", index=False)

        print("\nSaved", save_to)

        if download:
            try:
                from google.colab import files as colab_files
                colab_files.download(save_to)
            except Exception:
                pass

    return out

# %% ------------------------------------------------------------ 6. LIVE DETECTOR
class RansomwareEarlyWarning:
    """Real-time use. Call push(window_row) once per window with ONE row holding all telemetry
    columns of that window (dict / pandas Series). Each layer picks its own columns, the four layers
    score simultaneously, and the ensemble returns the weighted-average verdict."""
    def __init__(self, artifact=ARTIFACT, ensemble=None):
        self.specs, self.models = load_pretrained(artifact)
        self.ens = ensemble or ENSEMBLE
        self.reset()
    def reset(self):
        self.buf = []; self.hist = []; self.n = 0
    def push(self, row):
        self.n += 1
        r = dict(row); r["run_id"] = "__live__"
        r.setdefault("window_start_ms", (self.n - 1) * 5000); r.setdefault("window_end_ms", self.n * 5000)
        self.buf.append(r); d = pd.DataFrame(self.buf)
        ok = [l for l in FILES if all(c in d.columns for c in dict.fromkeys(META + LAYER_COLUMNS[l]))]
        for l in ok: RUNS[l]["__live__"] = build_runs(d, l)["__live__"]
        def one(l): return l, float(predict(self.models[l], self.specs[l].inputs(["__live__"]))[0])
        with ThreadPoolExecutor(max_workers=len(ok)) as ex: p = dict(ex.map(one, ok))
        pf = float(self.ens.fuse({l: np.array(p.get(l, 0.0)) for l in FILES}, {l: np.array(float(l in p)) for l in FILES}))
        self.hist.append(pf)
        alert = len(self.hist) >= self.ens.persistence and all(h >= self.ens.threshold for h in self.hist[-self.ens.persistence:])
        return {"window": self.n, "p_layers": p, "p_final": pf,
                "verdict": "MALICIOUS" if pf >= self.ens.threshold else "BENIGN", "alert": bool(alert)}

def replay_run(det, df, rid, win_s=5.0):
    """feed the windows of one run from a unified DataFrame through the detector, one at a time"""
    det.reset(); g = df[df.run_id == rid].sort_values("window_start_ms")
    print(f"\n--- live replay of {rid} ---")
    for t, (_, row) in enumerate(g.iterrows(), 1):
        o = det.push(row.to_dict())
        print(f"  t={t*win_s:4.0f}s  " + "  ".join(f"{l}={v:.2f}" for l, v in o["p_layers"].items())
              + f"  | FINAL={o['p_final']:.2f} {o['verdict']}{'  <-- ALERT' if o['alert'] else ''}")

# %% ------------------------------------------------------------ 6b. UNSEEN-SCENARIO TEST (optional)

# %% ------------------------------------------------------------ 6b. UNSEEN-SCENARIO TEST
def unseen_scenario_test():
    """Train without one malicious scenario and test on it plus held-out benign runs."""
    LABEL, SCEN = {}, {}

    for layer, fname in TRAIN_FILES.items():
        raw = pd.read_csv(os.path.join(DATA_DIR, fname))
        RUNS[layer] = build_runs(raw, layer)

        g = raw.groupby("run_id")[["label", "scenario"]].first()
        for rid, row in g.iterrows():
            LABEL[rid] = int(row["label"] == "malicious")
            SCEN.setdefault(rid, row["scenario"])

    common = sorted(set.intersection(*(set(RUNS[l]) for l in FILES)))
    benign = [r for r in common if LABEL[r] == 0]
    malicious = [r for r in common if LABEL[r] == 1]

    if not benign or not malicious:
        raise ValueError("Need both benign and malicious runs for this evaluation.")

    # Hold out benign runs, stratifying only when every group supports it.
    benign_scenarios = [SCEN[r] for r in benign]
    counts = Counter(benign_scenarios)
    can_stratify = len(counts) > 1 and min(counts.values()) >= 2

    _, benign_test = train_test_split(
        benign,
        test_size=0.2,
        random_state=SEED,
        stratify=benign_scenarios if can_stratify else None,
    )
    benign_test_set = set(benign_test)

    recalls, benign_correct = {}, {}

    for held in sorted({SCEN[r] for r in malicious}):
        held_ids = [r for r in malicious if SCEN[r] == held]
        held_set = set(held_ids)

        # Explicitly exclude the held scenario and held-out benign runs.
        pool = [
            r for r in common
            if r not in held_set and r not in benign_test_set
        ]

        if not any(LABEL[r] == 1 for r in pool):
            print(f"Skipping {held}: no malicious runs remain for training.")
            continue

        pool_scenarios = [SCEN[r] for r in pool]
        pool_counts = Counter(pool_scenarios)
        stratify_pool = (
            pool_scenarios
            if len(pool_counts) > 1 and min(pool_counts.values()) >= 2
            else None
        )

        tr, va = train_test_split(
            pool,
            test_size=0.15,
            random_state=SEED,
            stratify=stratify_pool,
        )

        # Safety assertions: no held malicious scenario in train or validation.
        assert not (set(tr) & held_set)
        assert not (set(va) & held_set)

        test_ids = held_ids + list(benign_test)
        predictions = {}

        print(
            f"\n--- held-out scenario: {held} | "
            f"train={len(tr)} validation={len(va)} "
            f"test={len(test_ids)} ---"
        )

        for layer in FILES:
            set_seed(SEED)
            spec = SPECS[layer]()
            spec.fit_scaler(tr)
            model = fit(spec, tr, va, TRAIN_KS, LABEL)
            predictions[layer] = predict(model, spec.inputs(test_ids))

        predictions["ENSEMBLE"] = ENSEMBLE.fuse(
            {layer: predictions[layer] for layer in FILES}
        )

        is_held = np.array([rid in held_set for rid in test_ids])

        recalls[held] = {
            layer: float((prob[is_held] >= ENSEMBLE.threshold).mean())
            for layer, prob in predictions.items()
        }
        benign_correct[held] = {
            layer: float((prob[~is_held] < ENSEMBLE.threshold).mean())
            for layer, prob in predictions.items()
        }

    recall_df = pd.DataFrame(recalls).T
    benign_df = pd.DataFrame(benign_correct).T

    print("\nUNSEEN-SCENARIO RECALL")
    print(recall_df.round(3).to_string())
    print("\nBENIGN RUNS CLASSIFIED CORRECTLY")
    print(benign_df.round(3).to_string())

    return recall_df, benign_df


# %% ------------------------------------------------------------ 7. RUN
print_model_summary()
print_layer_columns()
if RUN_TRAINING:
    train_and_save()
if RUN_INFERENCE:
    results = test_dataset(TEST_FILE)
    det = RansomwareEarlyWarning()
    rd = results["run_df"]                      # demo: replay the longest run of each scenario as if live
    if "scenario" in rd:
        demo = [g.sort_values("windows", ascending=False).run_id.iloc[0] for _, g in rd.groupby("scenario")]
    else:
        demo = list(rd.sort_values("windows", ascending=False).run_id.iloc[:3])
    for rid in demo:
        replay_run(det, results["df"], rid, results["win_s"])
if RUN_UNSEEN_TEST:
    unseen_rec, unseen_ben = unseen_scenario_test()
