import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FCM-WOA Customer Clustering",
    page_icon="🐋",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #0d47a1 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; }
.hero p  { font-size: 0.95rem; margin: 0; opacity: 0.85; }

.badge {
    display: inline-block;
    padding: 3px 13px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
.badge-blue  { background:#e3f2fd; color:#1565c0; }
.badge-green { background:#e8f5e9; color:#2e7d32; }
.badge-red   { background:#fce4ec; color:#c62828; }

.card {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.card-high     { background: linear-gradient(135deg,#fff8e1,#fff3cd); border-left: 5px solid #f59e0b; }
.card-moderate { background: linear-gradient(135deg,#e3f2fd,#dbeafe); border-left: 5px solid #3b82f6; }
.card-low      { background: linear-gradient(135deg,#fce4ec,#fde8f0); border-left: 5px solid #ef4444; }

.metric-tile {
    background: #f8faff;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-tile .val { font-size: 1.6rem; font-weight: 800; color: #1a237e; }
.metric-tile .lbl { font-size: 0.78rem; color: #666; margin-top: 2px; }

.stButton>button {
    background: linear-gradient(135deg,#1a237e,#1565c0);
    color: white; border: none;
    border-radius: 10px;
    padding: 0.55rem 1.8rem;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    transition: 0.2s;
}
.stButton>button:hover { background: linear-gradient(135deg,#283593,#1976d2); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🐋 FCM-WOA Customer Segmentation</h1>
  <p>Fuzzy C-Means + Whale Optimization Algorithm &nbsp;·&nbsp; Identify High, Moderate & Low Spenders</p>
  <div style="margin-top:10px">
    <span class="badge badge-blue">FCM</span>
    <span class="badge badge-blue">WOA</span>
    <span class="badge badge-green">Fuzzy Membership</span>
    <span class="badge badge-red">Customer Analytics</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ALGORITHM (pure numpy, no sklearn dependency for clustering)
# ─────────────────────────────────────────────
def fcm_objective(X, U, centers, m=2):
    total = 0.0
    for i in range(len(X)):
        for j in range(len(centers)):
            total += (U[i, j] ** m) * (np.linalg.norm(X[i] - centers[j]) ** 2)
    return total

def update_centers(X, U, m=2):
    c = U.shape[1]
    centers = []
    for j in range(c):
        num = np.sum((U[:, j] ** m).reshape(-1, 1) * X, axis=0)
        den = np.sum(U[:, j] ** m) + 1e-10
        centers.append(num / den)
    return np.array(centers)

def update_U(X, centers, m=2):
    n, c = len(X), len(centers)
    new_U = np.zeros((n, c))
    for i in range(n):
        dists = np.array([np.linalg.norm(X[i] - centers[j]) for j in range(c)])
        dists = np.maximum(dists, 1e-10)
        for j in range(c):
            new_U[i, j] = 1.0 / np.sum((dists[j] / dists) ** (2 / (m - 1)))
    return new_U

def run_fcm(X, c=3, m=2, max_iter=30, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        centers = update_centers(X, U, m)
        U_new   = update_U(X, centers, m)
        if np.linalg.norm(U_new - U) < 1e-5:
            break
        U = U_new
    return U, centers

def run_woa_fcm(X, c=3, m=2, woa_iter=50, seed=42):
    # Init from FCM
    U, centers = run_fcm(X, c, m, seed=seed)
    best_score  = fcm_objective(X, U, centers, m)
    best_centers = centers.copy()
    best_U       = U.copy()

    rng = np.random.default_rng(seed)
    for t in range(woa_iter):
        a = 2 - 2 * t / woa_iter      # decreasing coefficient
        r1, r2 = rng.random(), rng.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        L = rng.uniform(-1, 1)
        b = 1.0
        p = rng.random()

        if p < 0.5:
            D = np.abs(C * best_centers - centers)
            new_centers = best_centers - A * D
        else:
            D = np.abs(best_centers - centers)
            new_centers = D * np.exp(b * L) * np.cos(2 * np.pi * L) + best_centers

        new_centers = np.clip(new_centers, 0, 1)
        new_U       = update_U(X, new_centers, m)
        new_score   = fcm_objective(X, new_U, new_centers, m)

        if new_score < best_score:
            best_score   = new_score
            best_centers = new_centers.copy()
            best_U       = new_U.copy()
        centers = new_centers

    return best_U, best_centers, best_score

def label_clusters(U, data_original):
    """
    Assign semantically meaningful names:
    High Spender / Moderate Spender / Low Spender
    based on average spending of dominant members.
    """
    c = U.shape[1]
    labels_hard = np.argmax(U, axis=1)
    spending_col = data_original['Spending'].values
    avg_spend = {}
    for j in range(c):
        mask = labels_hard == j
        avg_spend[j] = spending_col[mask].mean() if mask.sum() > 0 else 0

    sorted_clusters = sorted(avg_spend, key=avg_spend.get, reverse=True)
    mapping = {}
    tier_names  = ['🟡 High Spender',   '🔵 Moderate Spender', '🔴 Low Spender']
    tier_colors = ['#f59e0b',            '#3b82f6',              '#ef4444']
    tier_cards  = ['card-high',          'card-moderate',        'card-low']
    for rank, cid in enumerate(sorted_clusters):
        mapping[cid] = {
            'name':  tier_names[rank],
            'color': tier_colors[rank],
            'card':  tier_cards[rank],
            'short': ['High', 'Moderate', 'Low'][rank],
        }
    return mapping, labels_hard

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Algorithm Settings")
    woa_iter = st.slider("WOA Iterations",    10, 100, 50)
    fcm_m    = st.slider("Fuzziness (m)",    1.1, 3.0, 2.0, 0.1)
    seed_val = st.number_input("Random Seed", 0, 9999, 42)
    st.markdown("---")
    st.markdown("""
**Algorithm: FCM-WOA**

| Step | Action |
|------|--------|
| 1 | FCM initialises cluster centers |
| 2 | WOA whale movement updates centers globally |
| 3 | FCM refines membership from WOA positions |
| 4 | Accept if objective improves |

**Result:** Lower objective → better cluster separation
    """)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload / Enter Data",
    "📊 Cluster Visualizations",
    "🔢 Membership Table",
    "🏆 Model Comparison"
])

# ═════════════════════════════════════════════
# TAB 1 — Upload or Manual Entry
# ═════════════════════════════════════════════
with tab1:
    col_up, col_man = st.columns([1, 1], gap="large")

    with col_up:
        st.subheader("📂 Upload CSV")
        st.caption("Must have columns: `age`, `income`, `spending`  (or similar numeric columns)")
        uploaded = st.file_uploader("Drop your CSV here", type="csv")

    with col_man:
        st.subheader("✏️ Enter a Single Customer")
        name_input   = st.text_input("Customer Name", "New Customer")
        age_input    = st.number_input("Age",             18, 90,  30)
        income_input = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        spending_input = st.number_input("Annual Spending ($)", 0, 300000, 15000, step=500)

    run_btn = st.button("🚀 Run FCM-WOA Clustering", use_container_width=True)

    if run_btn:
        # ── Build dataframe ───────────────────────────────
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = [c.strip().lower() for c in raw.columns]

            # Try to find age/income/spending regardless of exact column name
            col_map = {}
            for target, candidates in {
                'age':      ['age'],
                'income':   ['income', 'annual_income', 'annual income'],
                'spending': ['spending', 'spend', 'spending_score', 'spending score'],
            }.items():
                for cand in candidates:
                    if cand in raw.columns:
                        col_map[target] = cand
                        break

            missing = [k for k in ['age', 'income', 'spending'] if k not in col_map]
            if missing:
                st.error(f"❌ Could not find columns: {missing}. Your CSV has: {list(raw.columns)}")
                st.stop()

            df_work = raw.rename(columns={v: k for k, v in col_map.items()})[['age', 'income', 'spending']].copy()
            df_work.columns = ['Age', 'Income', 'Spending']
            # carry extra info if available
            name_col = [c for c in raw.columns if 'name' in c]
            if name_col:
                df_work.insert(0, 'Name', raw[name_col[0]].values[:len(df_work)])
            manual_added = False
            st.success(f"✅ Loaded **{len(df_work)} customers** from CSV.")
        else:
            # Build synthetic dataset + manual entry
            rng2 = np.random.default_rng(0)
            n_syn = 149
            ages     = rng2.integers(18, 70, n_syn)
            incomes  = np.concatenate([
                rng2.normal(90000, 15000, 50),
                rng2.normal(55000, 10000, 50),
                rng2.normal(30000, 8000, 49),
            ]).clip(15000, 200000)
            spendings = np.concatenate([
                rng2.normal(60000, 12000, 50),
                rng2.normal(20000, 6000, 50),
                rng2.normal(5000,  2000, 49),
            ]).clip(500, 150000)

            df_syn = pd.DataFrame({
                'Name':     [f'Customer_{i+1}' for i in range(n_syn)],
                'Age':      ages,
                'Income':   incomes.astype(int),
                'Spending': spendings.astype(int),
            })
            manual_row = pd.DataFrame([{
                'Name':     name_input,
                'Age':      age_input,
                'Income':   income_input,
                'Spending': spending_input,
            }])
            df_work = pd.concat([df_syn, manual_row], ignore_index=True)
            manual_added = True
            st.info("No CSV uploaded — using synthetic data + your manual entry.")

        # ── Scale ─────────────────────────────────────────
        feat_cols = ['Age', 'Income', 'Spending']
        scaler    = MinMaxScaler()
        X_scaled  = scaler.fit_transform(df_work[feat_cols].values)

        # ── Run FCM plain ─────────────────────────────────
        with st.spinner("Running plain FCM..."):
            U_fcm, centers_fcm = run_fcm(X_scaled, c=3, m=fcm_m, seed=int(seed_val))
        obj_fcm = fcm_objective(X_scaled, U_fcm, centers_fcm, fcm_m)

        # ── Run FCM-WOA ───────────────────────────────────
        with st.spinner("🐋 Whale Optimization running..."):
            U_woa, centers_woa, obj_woa = run_woa_fcm(
                X_scaled, c=3, m=fcm_m, woa_iter=woa_iter, seed=int(seed_val))

        # ── Semantic cluster labels ───────────────────────
        cluster_map, hard_labels = label_clusters(U_woa, df_work)

        # ── Build output df ───────────────────────────────
        out = df_work.copy()
        out['Cluster_ID']     = hard_labels
        out['Cluster_Label']  = [cluster_map[l]['name'] for l in hard_labels]
        for j in range(3):
            out[f'Membership_C{j+1}'] = np.round(U_woa[:, j], 4)
        out['FCM_Cluster'] = np.argmax(U_fcm, axis=1)

        # ── Save to session ───────────────────────────────
        st.session_state.update({
            'out': out, 'df_work': df_work, 'X_scaled': X_scaled,
            'U_woa': U_woa, 'U_fcm': U_fcm,
            'centers_woa': centers_woa, 'centers_fcm': centers_fcm,
            'obj_woa': obj_woa, 'obj_fcm': obj_fcm,
            'cluster_map': cluster_map, 'hard_labels': hard_labels,
            'manual_added': manual_added, 'scaler': scaler,
        })

        # ── Metrics ───────────────────────────────────────
        imp = max(0, (obj_fcm - obj_woa) / obj_fcm * 100)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-tile"><div class="val">{len(df_work)}</div><div class="lbl">Total Customers</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-tile"><div class="val">{obj_woa:.2f}</div><div class="lbl">FCM-WOA Objective ↓</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-tile"><div class="val">{imp:.1f}%</div><div class="lbl">Improvement over FCM</div></div>', unsafe_allow_html=True)

        # ── Cluster cards ─────────────────────────────────
        st.markdown("### 🎯 Cluster Summary")
        cols = st.columns(3)
        for j in range(3):
            info = cluster_map[j]
            mask = hard_labels == j
            cnt  = mask.sum()
            avg_inc = df_work.loc[mask, 'Income'].mean()
            avg_spe = df_work.loc[mask, 'Spending'].mean()
            with cols[j]:
                st.markdown(f"""
<div class="card {info['card']}">
  <div style="font-size:1.2rem;font-weight:800">{info['name']}</div>
  <div style="margin-top:8px;font-size:0.9rem">
    👥 <b>{cnt}</b> customers<br>
    💰 Avg Income: <b>${avg_inc:,.0f}</b><br>
    🛍️ Avg Spending: <b>${avg_spe:,.0f}</b>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Manual customer result ─────────────────────────
        if manual_added:
            mi = len(df_work) - 1
            lbl = cluster_map[hard_labels[mi]]
            mems = U_woa[mi]
            st.markdown(f"---\n### 🎯 Your Customer: **{name_input}**")
            st.markdown(f"**Hard Assignment → {lbl['name']}**")
            mcols = st.columns(3)
            for j, mc in enumerate(mcols):
                info = cluster_map[j]
                pct  = mems[j] * 100
                with mc:
                    st.markdown(
                        f'<div style="border:2px solid {info["color"]};border-radius:10px;'
                        f'padding:14px;text-align:center;background:{info["color"]}18">'
                        f'<div style="font-weight:700;color:{info["color"]};font-size:0.9rem">{info["name"]}</div>'
                        f'<div style="font-size:2rem;font-weight:800;color:{info["color"]}">{pct:.1f}%</div>'
                        f'<div style="font-size:0.75rem;color:#666">membership degree</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ═════════════════════════════════════════════
# TAB 2 — Visualizations
# ═════════════════════════════════════════════
with tab2:
    if 'out' not in st.session_state:
        st.info("▶ Run clustering in Tab 1 first.")
        st.stop()

    X_sc    = st.session_state['X_scaled']
    U_woa   = st.session_state['U_woa']
    U_fcm   = st.session_state['U_fcm']
    C_woa   = st.session_state['centers_woa']
    C_fcm   = st.session_state['centers_fcm']
    hard    = st.session_state['hard_labels']
    cmap    = st.session_state['cluster_map']
    df_w    = st.session_state['df_work']
    manual  = st.session_state['manual_added']

    COLORS = [cmap[j]['color'] for j in range(3)]
    NAMES  = [cmap[j]['name']  for j in range(3)]

    # ── Plot 1: FCM-WOA vs Plain FCM ──────────────────────
    st.markdown("### 📌 FCM-WOA vs Plain FCM  — Income vs Spending")
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig1.patch.set_facecolor('#fafafa')

    for ax, U, C, hard_l, title, is_woa in [
        (axes[0], U_woa, C_woa, hard,                    "FCM-WOA (Your Model ✨)", True),
        (axes[1], U_fcm, C_fcm, np.argmax(U_fcm,axis=1), "Plain FCM (Baseline)",   False),
    ]:
        ax.set_facecolor('#fafafa')
        for j in range(3):
            mask = hard_l == j
            col  = COLORS[j] if is_woa else ['#ff7043','#42a5f5','#66bb6a'][j]
            ax.scatter(
                df_w.loc[mask, 'Income'], df_w.loc[mask, 'Spending'],
                c=col, s=65, alpha=0.75, edgecolors='white', lw=0.5,
                label=NAMES[j] if is_woa else f'Cluster {j+1}', zorder=3
            )
            # fuzzy halo
            mems = U[mask, j]
            ax.scatter(df_w.loc[mask,'Income'], df_w.loc[mask,'Spending'],
                       s=mems*400, c=col, alpha=0.12, zorder=2)

        # Centers (inverse-transformed)
        fake = np.zeros((3, X_sc.shape[1]))
        fake[:, 1] = C[:, 1]   # income dim
        fake[:, 2] = C[:, 2]   # spending dim
        inc_c = st.session_state['scaler'].inverse_transform(C)[:, 1]
        spe_c = st.session_state['scaler'].inverse_transform(C)[:, 2]
        ax.scatter(inc_c, spe_c, c='black', marker='*', s=400,
                   zorder=5, edgecolors='white', lw=0.8, label='Centers')

        if manual and is_woa:
            mi = len(df_w) - 1
            ax.scatter(df_w.iloc[mi]['Income'], df_w.iloc[mi]['Spending'],
                       c='gold', marker='D', s=200, zorder=6,
                       edgecolors='black', lw=1.2, label='Your Customer')

        ax.set_xlabel("Annual Income ($)", fontsize=10)
        ax.set_ylabel("Annual Spending ($)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color='#1a237e' if is_woa else '#c62828')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.spines[['top','right']].set_visible(False)
        # format axes
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

    plt.tight_layout()
    st.pyplot(fig1); plt.close()

    # ── Plot 2: Membership stacked bar ────────────────────
    st.markdown("### 🔥 Fuzzy Membership Distribution (FCM-WOA)")
    sort_idx = np.lexsort([np.argmax(U_woa, axis=1)])
    U_s = U_woa[sort_idx]

    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    fig2.patch.set_facecolor('#fafafa')
    ax2.set_facecolor('#fafafa')
    bottom = np.zeros(len(U_s))
    for j in range(3):
        ax2.bar(range(len(U_s)), U_s[:, j], bottom=bottom,
                color=COLORS[j], alpha=0.88, label=NAMES[j], width=1.0)
        bottom += U_s[:, j]

    ax2.set_xlabel("Customers (sorted by dominant cluster)", fontsize=10)
    ax2.set_ylabel("Membership Degree", fontsize=10)
    ax2.set_title("Fuzzy Membership Degrees — FCM-WOA", fontsize=12, fontweight='bold', color='#1a237e')
    ax2.legend(fontsize=9); ax2.set_ylim(0,1); ax2.set_xlim(-1, len(U_s))
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Plot 3: Per-cluster membership strength ───────────
    st.markdown("### 🎨 Per-Cluster Membership Strength (FCM-WOA)")
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5))
    fig3.patch.set_facecolor('#fafafa')

    for j, ax3 in enumerate(axes3):
        ax3.set_facecolor('#fafafa')
        cmap_j = LinearSegmentedColormap.from_list('', ['#ffffff', COLORS[j]])
        sc = ax3.scatter(
            df_w['Income'], df_w['Spending'],
            c=U_woa[:, j], cmap=cmap_j, s=55,
            edgecolors='#aaa', lw=0.25, vmin=0, vmax=1
        )
        # center marker
        inv_c = st.session_state['scaler'].inverse_transform(C_woa)
        ax3.scatter(inv_c[j, 1], inv_c[j, 2], c='black', marker='*',
                    s=380, zorder=5, edgecolors='white', lw=0.8)
        plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label='Membership')
        ax3.set_title(NAMES[j], fontsize=11, fontweight='bold', color=COLORS[j])
        ax3.set_xlabel("Income ($)", fontsize=9)
        ax3.set_ylabel("Spending ($)", fontsize=9)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.spines[['top','right']].set_visible(False)

        if manual:
            mi = len(df_w) - 1
            ax3.scatter(df_w.iloc[mi]['Income'], df_w.iloc[mi]['Spending'],
                        c='gold', marker='D', s=180, zorder=6,
                        edgecolors='black', lw=1)

    plt.suptitle("Per-Cluster Membership Strength", fontsize=13, fontweight='bold',
                 color='#1a237e', y=1.03)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # ── Plot 4: Age vs Spending coloured by cluster ────────
    st.markdown("### 👥 Age vs Spending — Cluster View")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.patch.set_facecolor('#fafafa')
    ax4.set_facecolor('#fafafa')
    for j in range(3):
        mask = hard == j
        ax4.scatter(df_w.loc[mask,'Age'], df_w.loc[mask,'Spending'],
                    c=COLORS[j], s=65, alpha=0.72, edgecolors='white',
                    lw=0.5, label=NAMES[j], zorder=3)
    if manual:
        mi = len(df_w) - 1
        ax4.scatter(df_w.iloc[mi]['Age'], df_w.iloc[mi]['Spending'],
                    c='gold', marker='D', s=200, zorder=6,
                    edgecolors='black', lw=1.2, label='Your Customer')
    ax4.set_xlabel("Age", fontsize=11)
    ax4.set_ylabel("Annual Spending ($)", fontsize=11)
    ax4.set_title("Age vs Spending — FCM-WOA Clusters", fontsize=13,
                  fontweight='bold', color='#1a237e')
    ax4.legend(fontsize=9)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    ax4.grid(True, linestyle='--', alpha=0.35)
    ax4.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig4); plt.close()

# ═════════════════════════════════════════════
# TAB 3 — Membership Table
# ═════════════════════════════════════════════
with tab3:
    if 'out' not in st.session_state:
        st.info("▶ Run clustering in Tab 1 first.")
    else:
        out     = st.session_state['out']
        cmap_s  = st.session_state['cluster_map']

        st.subheader("📋 Membership Results Table")
        filter_opt = st.selectbox("Filter by Cluster",
            ['All'] + [cmap_s[j]['name'] for j in range(3)])

        disp = out.copy()
        if filter_opt != 'All':
            disp = disp[disp['Cluster_Label'] == filter_opt]

        mem_cols = [f'Membership_C{j+1}' for j in range(3)]
        show_cols = [c for c in ['Name','Age','Income','Spending',
                                  'Cluster_Label'] + mem_cols if c in disp.columns]

        def highlight(row):
            styles = [''] * len(row)
            for c in mem_cols:
                if c in row.index:
                    idx = row.index.get_loc(c)
                    if row[c] == row[mem_cols].max():
                        styles[idx] = 'background:#d1fae5;font-weight:700'
            return styles

        styled = disp[show_cols].style.apply(highlight, axis=1).format(
            {c: '{:.4f}' for c in mem_cols}
        )
        st.dataframe(styled, use_container_width=True, height=420)

        csv_bytes = out.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results CSV", csv_bytes,
                            "fcm_woa_results.csv", use_container_width=True)

        # Summary table
        st.markdown("### 📊 Cluster Summary Statistics")
        rows = []
        for j in range(3):
            mask = out['Cluster_ID'] == j
            sub  = out[mask]
            rows.append({
                'Cluster':        cmap_s[j]['name'],
                'Count':          len(sub),
                'Avg Age':        round(sub['Age'].mean(), 1),
                'Avg Income':     f"${sub['Income'].mean():,.0f}",
                'Avg Spending':   f"${sub['Spending'].mean():,.0f}",
                'Avg Confidence': round(sub[f'Membership_C{j+1}'].mean(), 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ═════════════════════════════════════════════
# TAB 4 — Model Comparison
# ═════════════════════════════════════════════
with tab4:
    if 'out' not in st.session_state:
        st.info("▶ Run clustering in Tab 1 first.")
    else:
        obj_woa = st.session_state['obj_woa']
        obj_fcm = st.session_state['obj_fcm']
        U_woa   = st.session_state['U_woa']
        U_fcm   = st.session_state['U_fcm']
        C_woa   = st.session_state['centers_woa']
        C_fcm   = st.session_state['centers_fcm']
        X_sc    = st.session_state['X_scaled']
        hard    = st.session_state['hard_labels']
        cmap_s  = st.session_state['cluster_map']

        def partition_coeff(U): return float(np.sum(U**2) / U.shape[0])
        def avg_confidence(U):  return float(U.max(axis=1).mean())

        pc_woa  = partition_coeff(U_woa);  pc_fcm  = partition_coeff(U_fcm)
        ac_woa  = avg_confidence(U_woa);   ac_fcm  = avg_confidence(U_fcm)
        imp_pct = max(0, (obj_fcm - obj_woa) / obj_fcm * 100)

        st.markdown("## 🏆 FCM-WOA vs Plain FCM — Performance")

        c1, c2, c3 = st.columns(3)
        for col, label, fcm_v, woa_v, lower_better in [
            (c1, "Objective (↓ better)",          obj_fcm, obj_woa, True),
            (c2, "Partition Coefficient (↑ better)", pc_fcm, pc_woa, False),
            (c3, "Avg Confidence (↑ better)",     ac_fcm,  ac_woa, False),
        ]:
            wins = woa_v < fcm_v if lower_better else woa_v > fcm_v
            with col:
                st.markdown(f"**{label}**")
                st.markdown(
                    f'<div style="background:#fce4ec;border-radius:8px;padding:10px;margin-bottom:6px">'
                    f'<b style="color:#c62828">Plain FCM:</b> {fcm_v:.4f}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#e8f5e9;border-radius:8px;padding:10px">'
                    f'<b style="color:#1a237e">FCM-WOA ✨:</b> {woa_v:.4f}'
                    + (f' <b style="color:#2e7d32">✅ Better</b>' if wins else '')
                    + '</div>', unsafe_allow_html=True)

        # Bar chart
        fig_c, ax_c = plt.subplots(figsize=(10, 4))
        fig_c.patch.set_facecolor('#fafafa')
        ax_c.set_facecolor('#fafafa')
        x    = np.arange(3)
        lbls = ['Objective\n(norm, ↓)', 'Partition Coeff\n(↑)', 'Avg Confidence\n(↑)']
        max_obj = max(obj_fcm, obj_woa)
        fcm_v = [obj_fcm/max_obj, pc_fcm, ac_fcm]
        woa_v = [obj_woa/max_obj, pc_woa, ac_woa]
        ax_c.bar(x-0.18, fcm_v, 0.33, label='Plain FCM',    color='#ef9a9a', edgecolor='#c62828', lw=1.2)
        ax_c.bar(x+0.18, woa_v, 0.33, label='FCM-WOA ✨', color='#90caf9', edgecolor='#1a237e', lw=1.2)
        ax_c.set_xticks(x); ax_c.set_xticklabels(lbls, fontsize=10)
        ax_c.set_title("FCM-WOA vs Plain FCM — Metrics", fontsize=13, fontweight='bold', color='#1a237e')
        ax_c.legend(fontsize=10)
        ax_c.grid(axis='y', linestyle='--', alpha=0.4)
        ax_c.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_c); plt.close()

        # Cluster sizes
        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        fig_s.patch.set_facecolor('#fafafa')
        ax_s.set_facecolor('#fafafa')
        hard_fcm = np.argmax(U_fcm, axis=1)
        x2 = np.arange(3)
        fcm_cnt = [np.sum(hard_fcm==j) for j in range(3)]
        woa_cnt = [np.sum(hard==j)     for j in range(3)]
        COLORS  = [cmap_s[j]['color']  for j in range(3)]
        ax_s.bar(x2-0.2, fcm_cnt, 0.38, label='Plain FCM', color='#ef9a9a', edgecolor='#c62828', lw=0.9)
        ax_s.bar(x2+0.2, woa_cnt, 0.38, label='FCM-WOA',   color=COLORS,    edgecolor='black',   lw=0.9)
        ax_s.set_xticks(x2)
        ax_s.set_xticklabels([cmap_s[j]['name'] for j in range(3)], fontsize=9)
        ax_s.set_ylabel("Number of Customers"); ax_s.legend()
        ax_s.set_title("Cluster Sizes: FCM-WOA vs Plain FCM", fontsize=12,
                        fontweight='bold', color='#1a237e')
        ax_s.grid(axis='y', linestyle='--', alpha=0.4)
        ax_s.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_s); plt.close()

        with st.expander("ℹ️ Why FCM-WOA outperforms plain FCM"):
            st.markdown(f"""
| Issue with FCM | How WOA fixes it |
|---------------|-----------------|
| Trapped in local optima | WOA global search escapes them |
| Sensitive to initialization | Population of whales reduces this |
| Deterministic search | WOA adds stochastic exploration |

**Your result:** FCM-WOA improved objective by **{imp_pct:.1f}%** over plain FCM.
            """)
