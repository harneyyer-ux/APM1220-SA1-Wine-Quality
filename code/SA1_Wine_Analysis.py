"""
APM1220 SA1 - Multivariate Analysis of the Wine Quality Dataset 
Sections: A) Data prep & EDA  B) MANOVA  C) PCA  D) EFA
Run from the repository root:  python code/SA1_Wine_Analysis.py
Outputs: output/figures/*.png, output/tables/*.csv, output/results.json
"""
import json, warnings, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import factor_analyzer.factor_analyzer as _fa_mod
# compatibility shim: factor_analyzer 0.5.1 passes a keyword removed in newer scikit-learn
_orig_check = _fa_mod.check_array
def _check(X, *a, **kw):
    if "force_all_finite" in kw: kw["ensure_all_finite"] = kw.pop("force_all_finite")
    return _orig_check(X, *a, **kw)
_fa_mod.check_array = _check
warnings.filterwarnings("ignore")
np.random.seed(2026)
FIG, TAB = "output/figures/", "output/tables/"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
R = {}                                   # results store for the report
sns.set_theme(style="whitegrid", font_scale=0.9)
COL = {"Red": "#a4243b", "White": "#d8b44a"}

# ======================= A. DATA PREPARATION & EDA =======================
red = pd.read_csv("data/winequality-red.csv", sep=";")
white = pd.read_csv("data/winequality-white.csv", sep=";")
red["color"], white["color"] = "Red", "White"
df = pd.concat([red, white], ignore_index=True)
df.columns = [c.replace(" ", "_") for c in df.columns]
VARS = [c for c in df.columns if c not in ("quality", "color")]          # 11 physicochemical variables
LAB = {v: v.replace("_", " ") for v in VARS}

R["n_red"], R["n_white"], R["n_total"], R["n_vars"] = len(red), len(white), len(df), df.shape[1]
R["missing"] = int(df.isna().sum().sum())
R["dup_total"] = int(df.duplicated().sum())
R["dup_by_color"] = df[df.duplicated()].color.value_counts().to_dict()

desc = df[VARS].describe().T
desc["skew"], desc["kurtosis"] = df[VARS].skew(), df[VARS].kurt()
desc["CV_%"] = desc["std"] / desc["mean"] * 100
desc.round(3).to_csv(TAB + "descriptives_overall.csv")
df.groupby("color")[VARS].agg(["mean", "std"]).round(3).to_csv(TAB + "descriptives_by_color.csv")
R["desc"] = desc.round(4).to_dict("index")

# univariate outliers (Tukey 1.5xIQR and |z|>3.29)
uo = {}
for v in VARS:
    q1, q3 = df[v].quantile([.25, .75]); i = q3 - q1
    uo[v] = {"iqr": int(((df[v] < q1 - 1.5*i) | (df[v] > q3 + 1.5*i)).sum()),
             "z329": int((np.abs(stats.zscore(df[v])) > 3.29).sum())}
R["uni_out"] = uo
# multivariate outliers: Mahalanobis D2 on 11 vars, chi-square p < .001
X = df[VARS].values
d = X - X.mean(0)
md2 = np.sum(d @ np.linalg.inv(np.cov(X.T)) * d, axis=1)
crit = stats.chi2.ppf(.999, len(VARS)); df["md2_all"] = md2
R["mv_out_11"] = int((md2 > crit).sum()); R["mv_crit_11"] = float(crit)

# distributions
fig, ax = plt.subplots(3, 4, figsize=(13, 8.5)); ax = ax.ravel()
for a, v in zip(ax, VARS):
    sns.kdeplot(data=df, x=v, hue="color", palette=COL, fill=True, alpha=.4, common_norm=False, ax=a, legend=False)
    a.set_title(LAB[v], fontsize=9); a.set_xlabel(""); a.set_ylabel("")
ax[-1].axis("off")
ax[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=.5) for c in COL.values()], labels=list(COL.keys()), loc="center")
fig.suptitle("Figure 1. Density of physicochemical variables by wine color", y=1.0); plt.tight_layout()
plt.savefig(FIG + "fig1_density_by_color.png", dpi=150); plt.close()
# boxplots
fig, ax = plt.subplots(3, 4, figsize=(13, 8.5)); ax = ax.ravel()
for a, v in zip(ax, VARS):
    sns.boxplot(data=df, x="color", y=v, palette=COL, ax=a, fliersize=2)
    a.set_title(LAB[v], fontsize=9); a.set_xlabel(""); a.set_ylabel("")
ax[-1].axis("off"); fig.suptitle("Figure 2. Boxplots of physicochemical variables by wine color", y=1.0); plt.tight_layout()
plt.savefig(FIG + "fig2_boxplots_by_color.png", dpi=150); plt.close()
# correlation heatmap
C = df[VARS].corr(); C.round(3).to_csv(TAB + "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, annot_kws={"size": 7}, ax=ax,
            xticklabels=[LAB[v] for v in VARS], yticklabels=[LAB[v] for v in VARS])
ax.set_title("Figure 3. Pearson correlation matrix (combined data, n = 6,497)"); plt.tight_layout()
plt.savefig(FIG + "fig3_corr_heatmap.png", dpi=150); plt.close()
pairs = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().sort_values(key=abs, ascending=False)
R["top_corr"] = [(a, b, round(float(r), 3)) for (a, b), r in pairs.head(8).items()]
# scatterplot matrix for a key subset
sub = ["fixed_acidity", "volatile_acidity", "residual_sugar", "total_sulfur_dioxide", "density", "alcohol"]
g = sns.pairplot(df.sample(2000, random_state=1), vars=sub, hue="color", palette=COL,
                 plot_kws={"s": 6, "alpha": .5}, diag_kind="kde", height=1.6)
g.figure.suptitle("Figure 4. Scatterplot matrix of selected variables (random subsample, n = 2,000)", y=1.02)
g.savefig(FIG + "fig4_scatter_matrix.png", dpi=140); plt.close("all")

# ======================= B. MANOVA =======================
DV = ["fixed_acidity", "volatile_acidity", "residual_sugar", "chlorides", "total_sulfur_dioxide", "sulphates", "alcohol"]
R["dv"] = DV
Cdv = df[DV].corr()
off = Cdv.where(np.triu(np.ones(Cdv.shape), 1).astype(bool)).stack()
R["dv_corr_range"] = [float(off.min()), float(off.max())]; R["dv_det"] = float(np.linalg.det(Cdv.values))
# (2) multivariate normality: Henze-Zirkler per group + Mahalanobis chi-square QQ
mvn = {}
for c in ["Red", "White"]:
    r = pg.multivariate_normality(df.loc[df.color == c, DV], alpha=.05); mvn[c] = {"HZ": float(r[0]), "p": float(r[1])}
R["mvn"] = mvn
# (3) multivariate outliers within each group (Mahalanobis D2, p<.001)
mo = {}; df["md2_grp"] = np.nan
for c in ["Red", "White"]:
    m = df.color == c; Z = df.loc[m, DV].values; dd = Z - Z.mean(0)
    d2 = np.sum(dd @ np.linalg.inv(np.cov(Z.T)) * dd, axis=1); df.loc[m, "md2_grp"] = d2
    mo[c] = {"n": int(m.sum()), "outliers": int((d2 > stats.chi2.ppf(.999, len(DV))).sum())}
R["mv_out_dv"] = mo; R["mv_crit_dv"] = float(stats.chi2.ppf(.999, len(DV)))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, c in zip(ax, ["Red", "White"]):
    d2 = np.sort(df.loc[df.color == c, "md2_grp"].values); n_ = len(d2)
    th = stats.chi2.ppf((np.arange(1, n_ + 1) - .5) / n_, len(DV))
    a.scatter(th, d2, s=5, color=COL[c]); a.plot([0, th.max()], [0, th.max()], "k--", lw=1)
    a.set_title(f"{c} wines (n = {n_})"); a.set_xlabel(r"Theoretical $\chi^2_{7}$ quantiles"); a.set_ylabel("Mahalanobis $D^2$")
fig.suptitle("Figure 5. Chi-square Q-Q plots of Mahalanobis distances (MANOVA dependent variables)"); plt.tight_layout()
plt.savefig(FIG + "fig5_mvn_qq.png", dpi=150); plt.close()
# (4) homogeneity of covariance: Box's M, Levene
Sg = {c: np.cov(df.loc[df.color == c, DV].T) for c in ["Red", "White"]}; ng = {c: int((df.color == c).sum()) for c in Sg}
Sp = sum((ng[c] - 1) * Sg[c] for c in Sg) / (len(df) - 2)
M = (len(df) - 2) * np.linalg.slogdet(Sp)[1] - sum((ng[c] - 1) * np.linalg.slogdet(Sg[c])[1] for c in Sg)
pdv = len(DV); cfac = (sum(1 / (ng[c] - 1) for c in Sg) - 1 / (len(df) - 2)) * (2 * pdv**2 + 3 * pdv - 1) / (6 * (pdv + 1) * (2 - 1))
chiM = M * (1 - cfac); dfM = pdv * (pdv + 1) / 2
R["boxm"] = {"M": float(M), "chi2": float(chiM), "df": float(dfM), "p": float(stats.chi2.sf(chiM, dfM))}
lev = {}
for v in DV:
    W_, p_ = stats.levene(df.loc[df.color == "Red", v], df.loc[df.color == "White", v], center="median"); lev[v] = {"W": float(W_), "p": float(p_)}
R["levene"] = lev
R["logdet"] = {c: float(np.linalg.slogdet(np.cov(df.loc[df.color == c, DV].T))[1]) for c in ["Red", "White"]}

def run_manova(d_):
    mv = MANOVA.from_formula(" + ".join(DV) + " ~ color", data=d_).mv_test()["color"]["stat"]
    out = {}
    for k in ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]:
        r = mv.loc[k]; out[k] = {"value": float(r["Value"]), "F": float(r["F Value"]), "df1": float(r["Num DF"]), "df2": float(r["Den DF"]), "p": float(r["Pr > F"])}
    return out
R["manova"] = run_manova(df)
R["partial_eta2_multivariate"] = 1 - R["manova"]["Wilks' lambda"]["value"]      # s = 1 -> 1 - Lambda
# sensitivity analyses
d_nodup = df.drop_duplicates(subset=VARS + ["color"]); R["n_nodup"] = len(d_nodup); R["manova_nodup"] = run_manova(d_nodup)
d_noout = df[df.md2_grp <= R["mv_crit_dv"]]; R["n_noout"] = len(d_noout); R["manova_noout"] = run_manova(d_noout)
# follow-up univariate ANOVA, Welch, eta2, Cohen's d, Bonferroni
fu = []
for v in DV:
    a, b = df.loc[df.color == "Red", v], df.loc[df.color == "White", v]
    F, p_ = stats.f_oneway(a, b); wt = stats.ttest_ind(a, b, equal_var=False)
    ssb = len(a) * (a.mean() - df[v].mean())**2 + len(b) * (b.mean() - df[v].mean())**2; sst = ((df[v] - df[v].mean())**2).sum()
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    fu.append({"variable": v, "M_red": a.mean(), "SD_red": a.std(), "M_white": b.mean(), "SD_white": b.std(), "F": F, "p": p_,
               "p_bonf": min(1, p_ * len(DV)), "eta2": ssb / sst, "d": (a.mean() - b.mean()) / sp,
               "Welch_t": float(wt.statistic), "Welch_df": float(wt.df), "Welch_p": float(wt.pvalue)})
fu = pd.DataFrame(fu); fu.to_csv(TAB + "manova_followup_anova.csv", index=False); R["followup"] = fu.to_dict("records")
# discriminant function (single function, s = 1): standardized coefficients & structure correlations
Z = df[DV]
Sw = sum(((Z[df.color == c] - Z[df.color == c].mean()).T @ (Z[df.color == c] - Z[df.color == c].mean())).values for c in ["Red", "White"]) / (len(df) - 2)
dm = (Z[df.color == "Red"].mean() - Z[df.color == "White"].mean()).values
w = np.linalg.solve(Sw, dm); std_coef = w * np.sqrt(np.diag(Sw)); std_coef /= np.sqrt((std_coef**2).sum())
score = Z.values @ w; struct = [np.corrcoef(Z[v], score)[0, 1] for v in DV]
R["lda"] = {"std_coef": dict(zip(DV, np.round(std_coef, 3).tolist())), "structure": dict(zip(DV, np.round(struct, 3).tolist())), "mahal_D2": float(dm @ w)}
fig, ax = plt.subplots(figsize=(7.5, 4))
sns.kdeplot(x=score, hue=df.color, palette=COL, fill=True, alpha=.5, common_norm=False, ax=ax)
ax.set_xlabel("Discriminant score (linear combination of the 7 DVs)")
ax.set_title("Figure 6. Separation of red and white wines on the discriminant function"); plt.tight_layout()
plt.savefig(FIG + "fig6_discriminant.png", dpi=150); plt.close()

# ======================= C. PCA =======================
Xs = (df[VARS] - df[VARS].mean()) / df[VARS].std()
Cx = df[VARS].corr().values; n, p = Xs.shape
R["bartlett"] = dict(zip(["chi2", "p"], map(float, calculate_bartlett_sphericity(df[VARS]))))
kmo_v, kmo_t = calculate_kmo(df[VARS]); R["kmo_total"] = float(kmo_t); R["kmo_var"] = dict(zip(VARS, np.round(kmo_v, 3).tolist()))
R["det_corr"] = float(np.linalg.det(Cx)); R["vif"] = dict(zip(VARS, np.round(np.diag(np.linalg.inv(Cx)), 2).tolist()))
ev, evec = np.linalg.eigh(Cx); o = np.argsort(ev)[::-1]; ev, evec = ev[o], evec[:, o]
R["eig"] = ev.tolist(); R["pve"] = (ev / ev.sum()).tolist(); R["cpve"] = np.cumsum(ev / ev.sum()).tolist()
# Horn's parallel analysis (1000 random normal datasets, 95th percentile)
sims = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(np.random.randn(n, p).T)))[::-1] for _ in range(1000)])
pa95 = np.percentile(sims, 95, axis=0); R["pa95"] = pa95.tolist(); R["pa_retain"] = int((ev > pa95).sum())
k = np.arange(1, p + 1)
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(k, ev, "o-", color="#1f3b73", label="Observed eigenvalue"); ax.plot(k, pa95, "s--", color="gray", label="Parallel analysis (95th pct)")
ax.axhline(1, color="r", ls=":", lw=1, label="Kaiser (λ = 1)"); ax.set_xlabel("Component"); ax.set_ylabel("Eigenvalue"); ax.set_xticks(k); ax.legend()
ax.set_title("Figure 7. Scree plot with parallel analysis"); plt.tight_layout(); plt.savefig(FIG + "fig7_scree.png", dpi=150); plt.close()
NPC = 3
load = evec[:, :NPC] * np.sqrt(ev[:NPC])
for j in range(NPC):                                    # sign convention: largest |loading| positive
    if load[np.argmax(np.abs(load[:, j])), j] < 0: load[:, j] *= -1; evec[:, j] *= -1
L = pd.DataFrame(load, index=VARS, columns=[f"PC{i+1}" for i in range(NPC)]); L["communality"] = (L**2).sum(1)
L.round(3).to_csv(TAB + "pca_loadings.csv"); R["pca_load"] = L.round(3).to_dict("index"); R["npc"] = NPC
R["pc_cum"] = float(np.cumsum(ev / ev.sum())[NPC - 1])
R["pc4_load"] = dict(zip(VARS, np.round(evec[:, 3] * np.sqrt(ev[3]), 3).tolist()))
scores = Xs.values @ evec[:, :NPC]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, (i, j) in zip(ax, [(0, 1), (0, 2)]):
    a.scatter(scores[:, i], scores[:, j], c=df.color.map(COL), s=4, alpha=.35)
    for v, (x, y) in zip(VARS, load[:, [i, j]]):
        a.arrow(0, 0, x * 3.2, y * 3.2, color="k", head_width=.08, lw=.8); a.text(x * 3.5, y * 3.5, LAB[v], fontsize=7.5)
    a.axhline(0, color="gray", lw=.5); a.axvline(0, color="gray", lw=.5)
    a.set_xlabel(f"PC{i+1} ({ev[i]/p:.1%})"); a.set_ylabel(f"PC{j+1} ({ev[j]/p:.1%})")
ax[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=k_) for k_, c in COL.items()])
fig.suptitle("Figure 8. PCA biplots (points = wines by color; arrows = loadings, scaled)"); plt.tight_layout()
plt.savefig(FIG + "fig8_pca_biplot.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="gray"))
for v, (x, y) in zip(VARS, load[:, :2]):
    ax.arrow(0, 0, x, y, color="#1f3b73", head_width=.02); ax.text(x * 1.06, y * 1.06, LAB[v], fontsize=8)
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Figure 9. PCA loading plot (PC1 vs PC2)"); plt.tight_layout()
plt.savefig(FIG + "fig9_pca_loadings.png", dpi=150); plt.close()
R["pc_color_d"] = {f"PC{i+1}": float((scores[(df.color == "Red").values, i].mean() - scores[(df.color == "White").values, i].mean()) / scores[:, i].std()) for i in range(NPC)}

# ======================= D. EFA =======================
def fa(nf, method, rot, data=None):
    f = FactorAnalyzer(n_factors=nf, method=method, rotation=rot); f.fit(df[VARS] if data is None else data); return f
def reduced_eigs(Rm):                                   # eigenvalues of reduced matrix (SMC communalities on diagonal)
    h = 1 - 1 / np.diag(np.linalg.inv(Rm)); Rr = Rm.copy(); np.fill_diagonal(Rr, h); return np.sort(np.linalg.eigvalsh(Rr))[::-1]
rev = reduced_eigs(Cx); rsims = np.array([reduced_eigs(np.corrcoef(np.random.randn(n, p).T)) for _ in range(300)])
rpa = np.percentile(rsims, 95, axis=0); R["fa_reduced_eig"] = rev.tolist(); R["fa_reduced_pa"] = rpa.tolist(); R["fa_reduced_pa_retain"] = int((rev > rpa).sum())
def ml_fit(nf):
    f = fa(nf, "ml", None); Lm = f.loadings_; U = f.get_uniquenesses(); Sig = Lm @ Lm.T + np.diag(U)
    Fml = np.log(np.linalg.det(Sig)) + np.trace(Cx @ np.linalg.inv(Sig)) - np.log(np.linalg.det(Cx)) - p
    dof = ((p - nf)**2 - (p + nf)) / 2; chi = (n - 1 - (2 * p + 5) / 6 - 2 * nf / 3) * Fml
    rmsea = np.sqrt(max(Fml / dof - 1 / (n - 1), 0)) if dof > 0 else float("nan")
    return {"nf": nf, "chi2": float(chi), "df": float(dof), "rmsea": float(rmsea), "cum_var": float(f.get_factor_variance()[2][-1]), "min_uniq": float(U.min())}
R["ml_fit"] = [ml_fit(kk) for kk in range(1, 6)]
NF = 3
f_main = fa(NF, "ml", "oblimin"); f_ml = fa(NF, "minres", "oblimin"); f_vm = fa(NF, "ml", "varimax")   # f_ml = minres robustness check
def ftab(f):
    t = pd.DataFrame(f.loadings_, index=VARS, columns=[f"F{i+1}" for i in range(NF)]); t["h2"] = f.get_communalities(); t["u2"] = f.get_uniquenesses(); return t
T_main, T_ml, T_vm = ftab(f_main), ftab(f_ml), ftab(f_vm)
T_main.round(3).to_csv(TAB + "efa_loadings_ml_oblimin.csv"); T_ml.round(3).to_csv(TAB + "efa_loadings_minres_oblimin.csv"); T_vm.round(3).to_csv(TAB + "efa_loadings_ml_varimax.csv")
R["efa_main"] = T_main.round(3).to_dict("index"); R["efa_minres"] = T_ml.round(3).to_dict("index"); R["efa_vm"] = T_vm.round(3).to_dict("index")
R["efa_phi"] = np.round(f_main.phi_, 3).tolist(); R["efa_var_main"] = [np.round(x, 4).tolist() for x in f_main.get_factor_variance()]
R["efa_var_vm"] = [np.round(x, 4).tolist() for x in f_vm.get_factor_variance()]
def tucker(a, b): return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))
R["tucker_minres"] = [max(abs(tucker(T_main[f"F{i+1}"].values, T_ml[f"F{j+1}"].values)) for j in range(NF)) for i in range(NF)]
# model-implied fit of the main solution (ML): RMSR of residual correlations
Lm = f_main.loadings_; res = Cx - (Lm @ f_main.phi_ @ Lm.T + np.diag(f_main.get_uniquenesses()))
R["rmsr_main"] = float(np.sqrt(np.sum(res[np.triu_indices(p, 1)]**2) / (p * (p - 1) / 2)))
# ---- sensitivity: factorability / structure after removing the red-vs-white mean difference (pooled within-color) ----
Wd = df[VARS] - df.groupby("color")[VARS].transform("mean")
R["kmo_within"] = float(calculate_kmo(Wd)[1]); R["kmo_within_var"] = dict(zip(VARS, np.round(calculate_kmo(Wd)[0], 3).tolist()))
R["bartlett_within"] = dict(zip(["chi2", "p"], map(float, calculate_bartlett_sphericity(Wd))))
Cw = np.corrcoef(Wd.values.T); evw = np.sort(np.linalg.eigvalsh(Cw))[::-1]; R["eig_within"] = evw.tolist()
for c in ["Red", "White"]: R[f"kmo_{c}"] = float(calculate_kmo(df.loc[df.color == c, VARS])[1])
R["kmo_no_density"] = float(calculate_kmo(df[[v for v in VARS if v != "density"]])[1])
fw = FactorAnalyzer(n_factors=3, method="ml", rotation="oblimin"); fw.fit(Wd)
R["efa_within"] = pd.DataFrame(fw.loadings_, index=VARS, columns=["F1", "F2", "F3"]).round(3).to_dict("index")
# factor scores by color for the main solution (links EFA to MANOVA)
fs = f_main.transform(df[VARS]); R["fs_color_d"] = {f"F{i+1}": float((fs[(df.color == "Red").values, i].mean() - fs[(df.color == "White").values, i].mean()) / fs[:, i].std()) for i in range(NF)}
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(T_main[[f"F{i+1}" for i in range(NF)]], annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, yticklabels=[LAB[v] for v in VARS])
ax.set_title("Figure 10. EFA pattern loadings (ML extraction, oblimin rotation)"); plt.tight_layout(); plt.savefig(FIG + "fig10_efa_loadings.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(k, ev, "o-", label="Eigenvalue (correlation matrix)"); ax.plot(k, pa95, "s--", color="gray", label="Parallel analysis (95th pct)")
ax.plot(k, rev, "^-", color="#a4243b", label="Reduced-matrix eigenvalue"); ax.axhline(0, color="k", lw=.5); ax.axhline(1, color="r", ls=":", lw=1)
ax.set_xticks(k); ax.set_xlabel("Factor"); ax.set_ylabel("Eigenvalue"); ax.legend(fontsize=8); ax.set_title("Figure 11. Scree plot and parallel analysis for EFA")
plt.tight_layout(); plt.savefig(FIG + "fig11_efa_scree.png", dpi=150); plt.close()

json.dump(R, open("output/results.json", "w"), indent=1, default=float)
print("done")
