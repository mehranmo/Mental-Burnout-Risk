import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve, average_precision_score, confusion_matrix
import joblib

st.set_page_config(layout="wide", page_title="Burnout Risk")

NUM_COLS_BASE = ["age", "hours_social", "sleep_hours", "work_hours"]
CAT_COLS_BASE = ["gender"]

if "models" not in st.session_state: st.session_state.models = []
if "active_model_id" not in st.session_state: st.session_state.active_model_id = None
if "threshold" not in st.session_state: st.session_state.threshold = 0.5
if "last_prediction" not in st.session_state: st.session_state.last_prediction = {"risk": None}
if "data_version" not in st.session_state: st.session_state.data_version = "-"
if "last_train" not in st.session_state: st.session_state.last_train = None

@st.cache_data(show_spinner=False)
def load_default_csv():
    try: return pd.read_csv("Data/unified_with_productivity.csv")
    except Exception: return None

@st.cache_data(show_spinner=False)
def load_data(file):
    if file is None:
        df = load_default_csv()
        if df is None: return None, None
        b = df.to_csv(index=False).encode()
        return df, str(abs(hash(b)))[:8]
    b = file.read()
    df = pd.read_csv(io.BytesIO(b))
    return df, str(abs(hash(b)))[:8]

def normalize_schema(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {"burnout":"target","burnout_risk":"target","label":"target","risk":"target"}
    for a,b in alias.items():
        if a in df.columns and b not in df.columns: df = df.rename(columns={a:b})
    if "productivity" not in df.columns:
        guess = [c for c in df.columns if c.startswith("product") and c!="target"]
        if len(guess)==1: df = df.rename(columns={guess[0]:"productivity"})
    return df

def clean_and_filter(df, target_col):
    use = df[df[target_col].notna()].copy()
    use["gender"] = use["gender"].astype(str).str.strip().str.lower()
    for c in ["age","hours_social","sleep_hours","work_hours"]:
        if c in use.columns: use[c] = pd.to_numeric(use[c], errors="coerce")
    if "age" in use.columns: use["age"] = use["age"].clip(0,120)
    for c in ["hours_social","sleep_hours","work_hours"]:
        if c in use.columns: use[c] = use[c].clip(0,24)
    num_cols = [c for c in NUM_COLS_BASE if c in use.columns and not use[c].isna().all()]
    cat_cols = [c for c in CAT_COLS_BASE if c in use.columns]
    X = use[num_cols + cat_cols]
    y = use[target_col].astype(int).values
    return X, y, num_cols, cat_cols

def build_pipeline(num_cols, cat_cols, model_type, class_weight, calibration_method):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    if model_type=="Random Forest":
        base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight=("balanced" if class_weight else None))
    else:
        base = LogisticRegression(max_iter=500, class_weight=("balanced" if class_weight else None))
    if calibration_method=="Platt (sigmoid)":
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    elif calibration_method=="Isotonic":
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    else:
        clf = base
    return Pipeline([("prep", pre), ("clf", clf)])

def cv_scores(pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy","roc_auc":"roc_auc","f1":"f1"}
    out = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {k: float(np.mean(v)) for k,v in out.items() if k.startswith("test_")}

def get_active_model():
    if st.session_state.active_model_id is None: return None
    for m in st.session_state.models:
        if m["id"]==st.session_state.active_model_id: return m
    return None

def set_active_model(mid): st.session_state.active_model_id = mid
def proba_for_row(pipe, row): return float(pipe.predict_proba(row)[0,1])
def ensure_reasonable_threshold():
    if st.session_state.threshold < 0.1:
        st.warning("Threshold too low. Using 0.10.")
        st.session_state.threshold = 0.1

m1,m2,m3 = st.columns(3)
with m1: st.metric("Predicted Risk", "-" if st.session_state.last_prediction["risk"] is None else f"{st.session_state.last_prediction['risk']*100:.1f}%")
with m2:
    am = get_active_model()
    st.metric("Model", "-" if am is None else am.get("name","Model"))
with m3: st.metric("Data version", st.session_state.data_version)

predict_tab, train_tab, explain_tab, fairness_tab, models_tab, about_tab, batch_tab = st.tabs(["Predict","Train","Explain","Fairness","Models","About","Batch Scoring"])

with predict_tab:
    model = get_active_model()
    if model is None:
        st.info("Train a model first in the Train tab.")
    else:
        label_name = model.get("label_name","target")
        st.caption(f"Active target: {label_name}")
        with st.form("predict_form"):
            c1,c2,c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 0, 120, 25)
                sleep_h = st.number_input("Sleep hours", 0.0, 24.0, 7.0, 0.5)
            with c2:
                social_h = st.number_input("Social hours", 0.0, 24.0, 3.0, 0.5)
                work_h = st.number_input("Work or Study hours", 0.0, 24.0, 8.0, 0.5)
            with c3:
                gender = st.selectbox("Gender", ["male","female","other"])
            with st.expander("Advanced"):
                thr = st.slider("Decision threshold", 0.0, 1.0, float(st.session_state.threshold), 0.01)
            sub = st.form_submit_button("Predict")
        if sub:
            st.session_state.threshold = thr
            ensure_reasonable_threshold()
            pipe = model["pipeline"]
            row = pd.DataFrame([{"age":age,"gender":gender,"hours_social":social_h,"sleep_hours":sleep_h,"work_hours":work_h}])
            prob1 = proba_for_row(pipe, row)
            st.session_state.last_prediction = {"risk": prob1}
            left,right = st.columns([1,1])
            if label_name=="target":
                pred = int(prob1 >= st.session_state.threshold)
                st.success(f"Burnout risk: {prob1*100:.1f}% ({'At risk' if pred==1 else 'Not at risk'})")
                with left:
                    st.metric("Burnout risk", f"{prob1*100:.1f}%")
                    st.metric("Label", "At risk" if pred==1 else "Not at risk")
                    st.progress(prob1)
            else:
                pred = int(prob1 >= st.session_state.threshold)
                low_prod_risk = 1.0 - prob1
                st.success(f"Productivity: {prob1*100:.1f}% ({'Good productivity' if pred==1 else 'Low productivity'})")
                with left:
                    st.metric("Good productivity probability", f"{prob1*100:.1f}%")
                    st.metric("Low productivity risk", f"{low_prod_risk*100:.1f}%")
                    st.progress(low_prod_risk)
            with right:
                st.caption("Model card")
                a1,a2,a3 = st.columns(3)
                a1.metric("Model", model["name"])
                a2.metric("Data version", model["data_version"])
                a3.metric("Threshold", f"{st.session_state.threshold:.2f}")
                b1,b2,b3 = st.columns(3)
                b1.metric("CV Acc", f"{model['cv']['test_accuracy']:.3f}")
                b2.metric("CV ROC-AUC", f"{model['cv']['test_roc_auc']:.3f}")
                b3.metric("CV F1", f"{model['cv']['test_f1']:.3f}")
                st.caption(f"Target: {model.get('label_name','target')} • Trained: {model['trained_at']} • Rows: {model['rows']} • Calibration: {model['options'].get('calibration','None')} • Class weight: {model['options']['class_weight']}")
        st.markdown("#### Small changes that flip the decision")
        model = get_active_model()
        if model is not None:
            ensure_reasonable_threshold()
            pipe = model["pipeline"]
            label_name = model.get("label_name","target")
            current = pd.DataFrame([{
                "age": age if "age" in model["num_cols"] else 0,
                "gender": gender,
                "hours_social": social_h if "hours_social" in model["num_cols"] else 0,
                "sleep_hours": sleep_h if "sleep_hours" in model["num_cols"] else 0,
                "work_hours": work_h if "work_hours" in model["num_cols"] else 0
            }])
            base_p = proba_for_row(pipe, current)
            base_pred = int(base_p >= st.session_state.threshold)

            # required fix: goal depends on target
            desired_pred = 0 if label_name=="target" else 1

            if base_pred == desired_pred:
                st.info("You already meet the goal at the current threshold.")
            else:
                actionable = [c for c in ["sleep_hours","work_hours","hours_social"] if c in model["num_cols"]]
                if len(actionable)==0:
                    st.info("No adjustable features available for this model.")
                else:
                    steps = np.arange(-3.0, 3.5, 0.5)
                    best_flip, best_improve = None, None
                    def day_ok(row):
                        tot = sum(float(row.loc[0,k]) for k in ["sleep_hours","work_hours","hours_social"] if k in row.columns)
                        return 0 <= tot <= 24
                    for col in actionable:
                        base_val = float(current.loc[0,col])
                        for d in steps:
                            if d==0: continue
                            trial = current.copy()
                            trial.loc[0,col] = np.clip(base_val + d, 0, 24)
                            if not day_ok(trial): continue
                            p = proba_for_row(pipe, trial)
                            pred = int(p >= st.session_state.threshold)
                            change = abs(d)
                            if pred == desired_pred:
                                if best_flip is None or change < best_flip["change"]:
                                    best_flip = {"col":col,"d":d,"p":p,"change":change}
                            else:
                                improve = (base_p - p) if label_name=="target" else (p - base_p)
                                if best_improve is None or improve > best_improve["improve"] or (np.isclose(improve,best_improve["improve"]) and change < best_improve["change"]):
                                    best_improve = {"col":col,"d":d,"p":p,"improve":improve,"change":change}
                    if best_flip:
                        title = "New burnout risk" if label_name=="target" else "New good productivity"
                        st.success(f"Change {best_flip['col'].replace('_',' ')} by {best_flip['d']:+.1f} h. {title}: {best_flip['p']*100:.1f}% (decision flips).")
                    elif best_improve and best_improve["improve"]>0:
                        title = "New burnout risk" if label_name=="target" else "New good productivity"
                        st.info(f"Closest improvement: change {best_improve['col'].replace('_',' ')} by {best_improve['d']:+.1f} h. {title}: {best_improve['p']*100:.1f}%.")
                    else:
                        st.info("No helpful change found within ±3 hours.")

with train_tab:
    src = st.radio("Training data", ["Default","Upload"], horizontal=True, key="train_src")
    file = st.file_uploader("CSV for training", type=["csv"], key="train_csv") if src=="Upload" else None
    df, dv = load_data(file)
    if df is None:
        st.info("Provide a CSV or keep Default once available.")
    else:
        df = normalize_schema(df)
        target_choice = st.selectbox("Select target to predict", ["Burnout","Productivity"], key="target_sel")
        target_col = "target" if target_choice=="Burnout" else "productivity"
        if target_choice=="Productivity" and "productivity" not in df.columns:
            up_prod = st.file_uploader("Upload productivity CSV with columns: age, gender, hours_social, sleep_hours, work_hours, productivity", type=["csv"], key="prod_csv")
            if up_prod is not None:
                add = pd.read_csv(up_prod)
                add = normalize_schema(add)
                need = set(NUM_COLS_BASE + CAT_COLS_BASE + ["productivity"])
                if need.issubset(set(add.columns)):
                    df = pd.concat([df, add[NUM_COLS_BASE + CAT_COLS_BASE + ["productivity"]]], ignore_index=True)
                    df = normalize_schema(df)
                else:
                    st.error(f"Uploaded file missing columns. Needed: {sorted(list(need))}")
                    st.stop()
        if target_col not in df.columns:
            st.error(f"Column '{target_col}' not found. Available: {list(df.columns)}")
            st.stop()
        X, y, num_cols, cat_cols = clean_and_filter(df, target_col)
        st.caption(f"Rows for {target_choice}: {len(X)} | Features used: {len(num_cols)+len(cat_cols)}")
        with st.expander("EDA Snapshot"):
            st.write("Class balance:", pd.Series(y).value_counts(normalize=True))
            st.write("Missing values (subset):", df[df[target_col].notna()][num_cols + cat_cols].isna().sum())
            if len(num_cols)>0: st.bar_chart(df[df[target_col].notna()][num_cols])
        c0,c1,c2,c3 = st.columns(4)
        with c0: model_type = st.selectbox("Model", ["Logistic Regression","Random Forest"], key="model_type")
        with c1: test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05, key="test_split")
        with c2: random_state = st.number_input("Random state", 0, 9999, 42, key="rand_state")
        with c3: class_weight_flag = st.checkbox("Use class_weight='balanced'", value=False, key="cw_flag")
        calibration_method = st.selectbox("Calibration", ["None","Platt (sigmoid)","Isotonic"], key="cal_method")

        if st.button("Quick compare models"):
            combos = [("Logistic Regression","None"),
                      ("Logistic Regression","Platt (sigmoid)"),
                      ("Logistic Regression","Isotonic"),
                      ("Random Forest","None"),
                      ("Random Forest","Platt (sigmoid)"),
                      ("Random Forest","Isotonic")]
            rows = []
            for mt, cal in combos:
                p = build_pipeline(num_cols, cat_cols, mt, class_weight_flag, cal)
                s = cv_scores(p, X, y)
                rows.append({"model": mt, "calibration": cal, "acc": round(s["test_accuracy"],3), "roc_auc": round(s["test_roc_auc"],3), "f1": round(s["test_f1"],3)})
            res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False, ignore_index=True)
            st.dataframe(res, use_container_width=True)
            best = res.iloc[0]
            st.success(f"Best by ROC-AUC: {best['model']} + {best['calibration']} ({best['roc_auc']:.3f})")

        go = st.button("Train model", key="train_btn")
        if go:
            if len(np.unique(y))<2:
                st.error("Target has a single class. Provide data with both classes.")
            else:
                st.session_state.threshold = 0.5
                pipe = build_pipeline(num_cols, cat_cols, model_type, class_weight_flag, calibration_method)
                scores = cv_scores(pipe, X, y)
                Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
                pipe.fit(Xtr, ytr)
                proba = pipe.predict_proba(Xte)[:,1]
                preds = (proba >= st.session_state.threshold).astype(int)
                acc = accuracy_score(yte, preds)
                roc = roc_auc_score(yte, proba)
                f1 = f1_score(yte, preds)
                fpr,tpr,_ = roc_curve(yte, proba)
                pr_p,pr_r,_ = precision_recall_curve(yte, proba)
                ap = average_precision_score(yte, proba)
                cm = confusion_matrix(yte, preds)
                frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10)
                mid = f"M{len(st.session_state.models)+1}"
                eval_df = Xte.copy()
                eval_df["y_true"] = yte
                eval_df["proba"] = proba
                eval_df["pred"] = preds
                model_record = {
                    "id": mid,
                    "name": f"Model {len(st.session_state.models)+1}",
                    "pipeline": pipe,
                    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "data_version": dv,
                    "rows": int(len(X)),
                    "features": list(X.columns),
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                    "label_name": target_col,
                    "cv": scores,
                    "eval_df": eval_df,
                    "options": {"model_type": model_type, "calibration": calibration_method, "class_weight": ("balanced" if class_weight_flag else None)},
                }
                st.session_state.models.append(model_record)
                set_active_model(mid)
                st.session_state.data_version = dv
                st.session_state.last_train = {"acc": float(acc), "roc": float(roc), "f1": float(f1), "fpr": fpr, "tpr": tpr, "precision": pr_p, "recall": pr_r, "ap": float(ap), "cm": cm, "mean_pred": mean_pred, "frac_pos": frac_pos}

    res = st.session_state.last_train
    if res is not None:
        st.success(f"Held-out: acc {res['acc']:.3f} | roc_auc {res['roc']:.3f} | f1 {res['f1']:.3f}")
        g1,g2 = st.columns(2)
        with g1:
            fig1,ax1 = plt.subplots(figsize=(4,3))
            ax1.plot(res["fpr"], res["tpr"]); ax1.plot([0,1],[0,1], linestyle="--")
            ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC curve")
            plt.tight_layout(); st.pyplot(fig1, use_container_width=False)
        with g2:
            fig2,ax2 = plt.subplots(figsize=(4,3))
            ax2.plot(res["recall"], res["precision"])
            ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title(f"PR curve (AP={res['ap']:.3f})")
            plt.tight_layout(); st.pyplot(fig2, use_container_width=False)
        g3,g4 = st.columns(2)
        with g3:
            fig3,ax3 = plt.subplots(figsize=(4,3))
            ax3.imshow(res["cm"]); ax3.set_title("Confusion matrix")
            ax3.set_xticks([0,1]); ax3.set_yticks([0,1]); ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
            for (i,j),v in np.ndenumerate(res["cm"]): ax3.text(j,i,str(v),ha="center",va="center")
            plt.tight_layout(); st.pyplot(fig3, use_container_width=False)
        with g4:
            fig4,ax4 = plt.subplots(figsize=(4,3))
            ax4.plot(res["mean_pred"], res["frac_pos"], marker="o"); ax4.plot([0,1],[0,1], linestyle="--")
            ax4.set_xlabel("Mean predicted prob"); ax4.set_ylabel("Fraction positive"); ax4.set_title("Calibration curve")
            plt.tight_layout(); st.pyplot(fig4, use_container_width=False)
        with st.expander("Threshold helpers"):
            act = get_active_model()
            if act is not None and "eval_df" in act:
                yh = act["eval_df"]["y_true"].values
                ph = act["eval_df"]["proba"].values
                if st.button("Set threshold by Youden’s J (ROC)"):
                    fprh,tprh,thrh = roc_curve(yh, ph)
                    j = tprh - fprh
                    st.session_state.threshold = float(np.clip(thrh[np.argmax(j)], 0.05, 0.95))
                    st.success(f"Threshold set to {st.session_state.threshold:.2f}")
                if st.button("Set threshold by max F1"):
                    prec,rec,thr = precision_recall_curve(yh, ph)
                    f1s = 2*prec*rec/(prec+rec+1e-9)
                    best = int(np.argmax(f1s[:-1])) if len(f1s)>1 else 0
                    if len(thr)>0:
                        st.session_state.threshold = float(np.clip(thr[best], 0.05, 0.95))
                        st.success(f"Threshold set to {st.session_state.threshold:.2f}")
                    else:
                        st.info("Not enough positive predictions to set by F1.")

with explain_tab:
    model = get_active_model()
    if model is None:
        st.info("Train or select a model first.")
    else:
        pipe = model["pipeline"]
        clf_step = pipe.named_steps["clf"]
        base = clf_step
        if hasattr(clf_step,"base_estimator") and clf_step.base_estimator is not None: base = clf_step.base_estimator
        elif hasattr(clf_step,"estimator") and clf_step.estimator is not None: base = clf_step.estimator
        prep = pipe.named_steps["prep"]
        cat_names = []
        if hasattr(prep,"named_transformers_") and "cat" in prep.named_transformers_:
            cat_pipe = prep.named_transformers_["cat"]
            if hasattr(cat_pipe,"named_steps") and "ohe" in cat_pipe.named_steps:
                try:
                    ohe = cat_pipe.named_steps["ohe"]; cat_names = list(ohe.get_feature_names_out(model["cat_cols"]))
                except Exception: cat_names = []
        feature_names = model["num_cols"] + cat_names
        if hasattr(base,"coef_"):
            coefs = base.coef_.ravel()
            w = pd.DataFrame({"feature":feature_names,"weight":coefs}).sort_values("weight", ascending=False)
            st.dataframe(w, use_container_width=True)
        elif hasattr(base,"feature_importances_"):
            imps = base.feature_importances_
            w = pd.DataFrame({"feature":feature_names,"importance":imps}).sort_values("importance", ascending=False)
            st.dataframe(w, use_container_width=True)
        else:
            st.info("No explainability attributes available for this model.")

with fairness_tab:
    model = get_active_model()
    if model is None or "eval_df" not in model:
        st.info("Train a model to view fairness metrics.")
    else:
        df_eval = model["eval_df"].copy()
        thr = st.session_state.threshold
        g = df_eval["gender"].astype(str).str.lower()
        df_eval["group_gender"] = g.where(g.isin(["male","female"]), "other")
        bins = [0,19,29,39,49,200]
        labels = ["<20","20s","30s","40s","50+"]
        df_eval["group_age"] = pd.cut(pd.to_numeric(df_eval["age"], errors="coerce").fillna(0), bins=bins, labels=labels, include_lowest=True)
        def group_table(col):
            rows = []
            for gname,d in df_eval.groupby(col):
                if len(d)==0 or d["y_true"].nunique()<2: continue
                y,p = d["y_true"].values, d["proba"].values
                pred = (p >= thr).astype(int)
                tpr = ((pred==1)&(y==1)).sum()/max((y==1).sum(),1)
                fpr = ((pred==1)&(y==0)).sum()/max((y==0).sum(),1)
                auc = roc_auc_score(y,p); acc = accuracy_score(y,pred)
                rows.append({"group":str(gname),"n":len(d),"auc":round(auc,3),"tpr":round(tpr,3),"fpr":round(fpr,3),"acc":round(acc,3)})
            return pd.DataFrame(rows).sort_values("group")
        st.markdown("#### By gender")
        gtab = group_table("group_gender")
        if len(gtab)==0: st.info("Not enough variation to compute gender groups.")
        else:
            st.dataframe(gtab, use_container_width=True)
            st.caption(f"AUC gap: {gtab['auc'].max() - gtab['auc'].min():.3f} | TPR gap: {gtab['tpr'].max() - gtab['tpr'].min():.3f}")
        st.markdown("#### By age band")
        atab = group_table("group_age")
        if len(atab)==0: st.info("Not enough variation to compute age groups.")
        else:
            st.dataframe(atab, use_container_width=True)
            st.caption(f"AUC gap: {atab['auc'].max() - atab['auc'].min():.3f} | TPR gap: {atab['tpr'].max() - atab['tpr'].min():.3f}")

with models_tab:
    if len(st.session_state.models)==0:
        st.info("No models yet.")
    else:
        names = [f"{m['id']} | {m['name']} | {m['data_version']} | label={m.get('label_name','target')}" for m in st.session_state.models]
        idx = 0
        if st.session_state.active_model_id is not None:
            for i,m in enumerate(st.session_state.models):
                if m["id"]==st.session_state.active_model_id: idx=i; break
        choice = st.radio("Select active model", names, index=idx)
        picked = st.session_state.models[names.index(choice)]
        set_active_model(picked["id"])
        st.session_state.data_version = picked["data_version"]
        st.session_state.threshold = st.session_state.get("threshold", 0.5)
        bio = io.BytesIO(); joblib.dump(picked, bio); bio.seek(0)
        st.download_button("Download model", data=bio, file_name=f"{picked['id']}.joblib")
        with st.expander("Active model card"):
            meta = {"trained":picked["trained_at"],"Data version":picked["data_version"],"rows":picked["rows"],"features":picked["features"],
                    "cv_accuracy":round(picked["cv"]["test_accuracy"],3),"cv_roc_auc":round(picked["cv"]["test_roc_auc"],3),
                    "cv_f1":round(picked["cv"]["test_f1"],3),"threshold":round(st.session_state.threshold,2),
                    "target":picked.get("label_name","target"),"options":picked["options"],"fairness_note":"Check Fairness tab for group metrics."}
            st.json(meta)

with about_tab:
    st.markdown("### About")
    st.markdown("This app predicts burnout risk or productivity using age, gender, social media hours, sleep hours, and work or study hours.")
    st.markdown("This is for learning and screening, not diagnosis.")

with batch_tab:
    st.markdown("### Batch Scoring")
    model = get_active_model()
    if model is None:
        st.info("Train or select a model first.")
    else:
        up = st.file_uploader("Upload CSV with columns: age, gender, hours_social, sleep_hours, work_hours", type=["csv"], key="batch_csv")
        if up is not None:
            df_in = pd.read_csv(up)

            if "gender" in df_in.columns:
                df_in["gender"] = df_in["gender"].astype(str).str.strip().str.lower()
            for c in ["age","hours_social","sleep_hours","work_hours"]:
                if c in df_in.columns:
                    df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
            if "age" in df_in.columns:
                df_in["age"] = df_in["age"].clip(0,120)
            for c in ["hours_social","sleep_hours","work_hours"]:
                if c in df_in.columns:
                    df_in[c] = df_in[c].clip(0,24)

            req = model["num_cols"] + model["cat_cols"]
            missing = [c for c in req if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                try:
                    pipe = model["pipeline"]
                    probs = pipe.predict_proba(df_in[req])[:,1]
                    preds = (probs >= st.session_state.threshold).astype(int)
                    df_out = df_in.copy()
                    label_name = model.get("label_name","target")
                    df_out[f"{label_name}_prob"] = probs
                    df_out[f"{label_name}_pred"] = preds
                    st.write("Scored sample:", df_out.head())
                    csv = df_out.to_csv(index=False).encode()
                    st.download_button("Download results", data=csv, file_name="batch_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error scoring batch: {e}")
