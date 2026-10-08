"""FraudShield AI V4 — portfolio transaction intelligence command center."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             confusion_matrix, precision_recall_curve)
from src.fraudshield.features import FEATURES, validate
from src.fraudshield.predict import load_model

st.set_page_config(page_title="FraudShield AI V4 | Command Center", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"]{font-family:'DM Sans',sans-serif}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 88% 0%,#182b3b 0%,#0b111d 35%,#080d16 90%);color:#e7eef7}
[data-testid="stSidebar"]{background:#101b2a;border-right:1px solid #293747}
h1,h2,h3{letter-spacing:-.03em;color:#f2f7fc!important}
p,li,label{color:#c8d3e0}
[data-testid="stMetric"]{background:linear-gradient(135deg,#142337,#101a29);border:1px solid #2a3b51;border-radius:14px;padding:16px}
[data-testid="stMetricValue"]{color:#eff9ff;font-weight:800}
[data-testid="stMetricLabel"]{color:#9fb1c6}
div.stButton>button[kind="primary"]{background:#13c3ae;color:#04120f;border:0;font-weight:700}
div.stButton>button{border-radius:10px}
[data-testid="stFileUploader"]{border-radius:12px}
[data-testid="stTabs"] button{font-size:14px;font-weight:700}
div[data-testid="stAlert"]{border-radius:12px}
a{color:#55e0ce}

/* V3: eliminate the Streamlit white header gap */
[data-testid="stHeader"], header[data-testid="stHeader"] {
  background: #0b1422 !important;
  color: #dceaf6 !important;
}
[data-testid="stToolbar"], [data-testid="stDecoration"] {background: transparent !important;}
[data-testid="stAppViewContainer"] > section {background: transparent !important;}
[data-testid="stSidebar"] {background: #0d1b2a !important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#dce7f3}
.sidebar-status {border:1px solid #1b675f;background:#10302f;color:#82f2d9;border-radius:10px;
padding:12px 14px;font-size:12px;font-weight:800;letter-spacing:.07em;margin:8px 0 16px}
.status-dot {display:inline-block;width:8px;height:8px;border-radius:50%;background:#24dfb6;
box-shadow:0 0 10px #24dfb6;margin-right:8px}
.sidebar-nav {color:#d6e6f2;line-height:2.65;font-size:14px;margin:8px 0 12px}
.sidebar-model {font-size:25px;font-weight:800;color:#eaf4ff;margin:5px 0 22px;letter-spacing:.02em}
.sidebar-threshold {font-size:24px;font-weight:750;color:#38e0c6;margin:4px 0 22px}
.sidebar-footer {font-size:11px;line-height:1.7;color:#90a4b9;border-top:1px solid #25384b;
padding-top:18px;margin-top:28px}
[data-testid="stFileUploader"] section, [data-testid="stExpander"] summary {
  background:#152438 !important;color:#e6f0fa !important;border-color:#2d4056 !important;
}
[data-testid="stFileUploader"] button {background:#243a50 !important;color:#f2f8ff !important}
[data-testid="stExpander"] summary p {color:#e6f0fa !important}
[data-testid="stTabs"] button {color:#c5d6e6 !important}
[data-testid="stTabs"] button[aria-selected="true"] {color:#4ce2ca !important}
[data-testid="stSidebar"] [role="radiogroup"] {gap: 6px !important;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    border:1px solid transparent;border-radius:10px;padding:9px 12px;
    background:transparent;cursor:pointer;transition:background .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#192e41 !important;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:#123e3d !important;border-color:#1a7d71 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {color:#e0edf8 !important;font-weight:600;}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stWidgetLabel"] {display:none;}
</style>
""", unsafe_allow_html=True)

COLORS={"teal":"#1bd4bd","amber":"#f4b85c","red":"#ff6b7b","blue":"#7da7ef"}
PLOT=dict(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#cbd7e6"),margin=dict(l=18,r=18,t=45,b=15))

@st.cache_resource(show_spinner=False)
def get_artifact():
    return load_model()

def score_frame(df, art):
    """Return a new frame, retaining Class only if explicitly supplied."""
    raw=art["model"].predict_proba(df[FEATURES])[:,1]
    result=df[["Time","Amount"]].copy()
    if "Class" in df.columns:result["Class"]=df["Class"].astype(int)
    result.insert(0,"Transaction_ID",np.arange(1,len(df)+1))
    result["risk_score"]=raw
    result["flagged"]=raw>=float(art["threshold"])
    result["risk_tier"]=np.select([result.risk_score>=art["threshold"],
                                   result.risk_score>=art["threshold"]*.5],
                                  ["HIGH","REVIEW"],default="LOW")
    return result

def chart(fig):
    fig.update_layout(**PLOT)
    fig.update_layout(
        title_font=dict(color="#f0f6ff",size=17,family="DM Sans, sans-serif"),
        font=dict(color="#d7e6f4",size=13),
        legend=dict(font=dict(color="#d7e6f4")),
        coloraxis_colorbar=dict(tickfont=dict(color="#d7e6f4"),title_font=dict(color="#d7e6f4"))
    )
    fig.update_xaxes(title_font=dict(color="#c9d9e9"),tickfont=dict(color="#b8cbe0"),
                     gridcolor="rgba(154,179,205,.13)")
    fig.update_yaxes(title_font=dict(color="#c9d9e9"),tickfont=dict(color="#b8cbe0"),
                     gridcolor="rgba(154,179,205,.13)")
    st.plotly_chart(fig,use_container_width=True)

def label_metrics(scored):
    if "Class" not in scored.columns:return
    y=scored["Class"].to_numpy(); flags=scored["flagged"].to_numpy()
    if len(np.unique(y))<2:
        st.info("Uploaded sample contains only one true class; PR-AUC is not defined.")
        return
    st.caption("Metrics below are for this uploaded sample, NOT held-out model performance.")
    a,b,c=st.columns(3)
    a.metric("Sample precision",f"{precision_score(y,flags,zero_division=0):.1%}")
    b.metric("Sample recall",f"{recall_score(y,flags,zero_division=0):.1%}")
    c.metric("Sample average precision",f"{average_precision_score(y,scored.risk_score):.3f}")
    cm=confusion_matrix(y,flags,labels=[0,1])
    chart(px.imshow(cm,text_auto=True,color_continuous_scale="Teal",
                    labels={"x":"Predicted class","y":"True class"},
                    x=["Genuine","Flagged"],y=["Genuine","Fraud"],
                    title="Uploaded sample — confusion matrix"))

def explain_transaction(art, record):
    """Best-effort SHAP for a single selected row; no invented explanation."""
    import shap
    pipeline=art["model"]
    if not hasattr(pipeline,"named_steps") or "model" not in pipeline.named_steps:
        raise ValueError("SHAP currently supports the selected fitted pipeline only.")
    pre=pipeline.named_steps.get("preprocess")
    estimator=pipeline.named_steps["model"]
    x=pd.DataFrame([record])[FEATURES]
    tx=pre.transform(x) if pre is not None else x.to_numpy()
    # ColumnTransformer moves Time and Amount first, then passthrough V1–V28.
    names=["Time","Amount"]+[f"V{i}" for i in range(1,29)]
    if hasattr(estimator,"get_booster"):
        explainer=shap.TreeExplainer(estimator)
        values=explainer.shap_values(tx)
    elif hasattr(estimator,"coef_"):
        explainer=shap.LinearExplainer(estimator,tx)
        values=explainer.shap_values(tx)
    else:
        raise ValueError("SHAP is unavailable for the selected estimator.")
    vals=np.asarray(values)
    if vals.ndim==3: vals=vals[:,:,1]
    contributions=pd.DataFrame({"feature":names,"shap_value":vals.reshape(-1)})
    return contributions.reindex(contributions.shap_value.abs().sort_values(ascending=False).index).head(12)

st.sidebar.markdown("## 🛡️ FraudShield")
st.sidebar.caption("AI / COMMAND CENTER · V4")
st.sidebar.markdown('<div class="sidebar-status"><span class="status-dot"></span> MODEL READY · LOCAL</div>', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("WORKSPACE")
page = st.sidebar.radio(
    "Navigate workspace",
    ["📊 Overview", "🔎 Investigation", "🧠 Model Lab", "▶ Simulator", "📡 Monitoring"],
    label_visibility="collapsed",
    key="fraudshield_page",
)
page = page.split(" ", 1)[1]
st.sidebar.divider()

st.caption("FRAUDSHIELD / INTELLIGENCE COMMAND CENTER · V4")
st.title("Transaction Risk Command Center")
st.markdown("**Explainable machine learning for suspicious transaction screening**")
st.caption("Anonymised ULB data · Model scores are uncalibrated, not real fraud probabilities.")

try:
    art=get_artifact()
except (FileNotFoundError, Exception) as exc:
    st.error(f"Model unavailable: {exc}. Train it using the README command.")
    st.stop()

report_path=Path("models/evaluation.json")
report=json.loads(report_path.read_text()) if report_path.exists() else {}
test=report.get("test_metrics",{})
st.sidebar.caption("ACTIVE MODEL")
st.sidebar.markdown(f'<div class="sidebar-model">{str(art.get("name","unknown")).upper()}</div>', unsafe_allow_html=True)
st.sidebar.caption("DECISION THRESHOLD")
st.sidebar.markdown(f'<div class="sidebar-threshold">{float(art["threshold"]):.4f}</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-footer">RESEARCH DEMO · LOCAL ONLY<br>Use anonymised data. Human review required.</div>', unsafe_allow_html=True)
if test:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Held-out PR-AUC",f"{test.get('average_precision',0):.3f}")
    c2.metric("Held-out recall",f"{test.get('recall',0):.1%}")
    c3.metric("Held-out precision",f"{test.get('precision',0):.1%}")
    c4.metric("Model threshold",f"{float(art['threshold']):.4f}")
    st.caption("Held-out metrics are from the original chronological test split; not live accuracy.")
else:
    st.info("Evaluation report not found. You can still use the trained model.")

with st.expander("📁 Upload transaction data",expanded=True):
    st.write("Upload the anonymised ULB CSV with Time, Amount and V1–V28. Optional Class labels enable sample evaluation.")
    uploaded=st.file_uploader("CSV dataset",type=["csv"],label_visibility="collapsed")
    if uploaded:
        try:
            raw=pd.read_csv(uploaded)
            if len(raw)>10000:raise ValueError("Demo limit: 10,000 rows per upload.")
            has_labels="Class" in raw.columns
            df=validate(raw,require_target=has_labels)
            if has_labels and not df["Class"].isin([0,1]).all():
                raise ValueError("Class labels must be 0 or 1.")
            scored=score_frame(df,art)
            st.session_state["fraudshield_data"]=df
            st.session_state["fraudshield_scored"]=scored
            st.success(f"{len(scored):,} transactions scored locally. Labels: {'present' if has_labels else 'not supplied'}.")
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")
df=st.session_state.get("fraudshield_data")
scored=st.session_state.get("fraudshield_scored")
if page == "Overview":
    st.subheader("Portfolio overview")
    if scored is None:
        st.info("Upload a CSV above to activate the transaction dashboards.")
    else:
        a,b,c,d=st.columns(4)
        a.metric("Scored",f"{len(scored):,}")
        b.metric("Flagged",f"{int(scored.flagged.sum()):,}")
        c.metric("Flag rate",f"{scored.flagged.mean():.1%}")
        d.metric("Average model score",f"{scored.risk_score.mean():.4f}")
        left,right=st.columns([1.5,1])
        with left:
            fig=px.histogram(scored,x="risk_score",nbins=45,title="Risk-score distribution",
                             color_discrete_sequence=[COLORS["teal"]])
            fig.add_vline(x=float(art["threshold"]),line_dash="dash",line_color=COLORS["amber"])
            chart(fig)
        with right:
            tiers=scored["risk_tier"].value_counts().rename_axis("tier").reset_index(name="count")
            chart(px.pie(tiers,names="tier",values="count",hole=.62,title="Screening tiers",
                         color="tier",color_discrete_map={"HIGH":COLORS["red"],"REVIEW":COLORS["amber"],"LOW":COLORS["teal"]}))
        st.dataframe(scored.sort_values("risk_score",ascending=False).head(200),
                     use_container_width=True,hide_index=True)
        st.download_button("⬇ Download all scored transactions",scored.to_csv(index=False),
                           "fraudshield_v2_scored.csv","text/csv")
        if "Class" in scored.columns:
            with st.expander("Evaluate against uploaded labels"):
                label_metrics(scored)
if page == "Investigation":
    st.subheader("Fraud investigation workbench")
    if scored is None:
        st.info("Upload a CSV to inspect individual transactions.")
    else:
        only_flagged=st.checkbox("Show flagged transactions only",value=True)
        view=scored[scored.flagged] if only_flagged else scored
        if view.empty:st.warning("No transactions match the filter. Uncheck 'flagged only'.")
        else:
            st.dataframe(view.sort_values("risk_score",ascending=False).head(250),
                         use_container_width=True,hide_index=True)
            ids=view.sort_values("risk_score",ascending=False)["Transaction_ID"].astype(int).tolist()
            selected=st.selectbox("Investigate transaction ID",ids)
            row=scored.loc[scored.Transaction_ID==selected].iloc[0]
            st.metric("Selected transaction model score",f"{row.risk_score:.4f}")
            st.write(f"**Decision:** {'FLAG FOR REVIEW' if row.flagged else 'Not flagged'} · **Amount:** {row.Amount:.2f} · **Time:** {row.Time:.0f}")
            if "Class" in scored.columns:
                st.write(f"**Known label in uploaded data:** {'Fraud' if row.Class==1 else 'Genuine'}")
            st.caption("A flag is a review recommendation, not proof of fraud.")
            if st.button("Explain this prediction with SHAP",type="primary"):
                with st.spinner("Calculating feature contributions..."):
                    try:
                        record=df.iloc[int(selected)-1][FEATURES].to_dict()
                        explain=explain_transaction(art,record)
                        fig=px.bar(explain.sort_values("shap_value"),x="shap_value",y="feature",
                                   orientation="h",title="SHAP contributions — model output scale",
                                   color="shap_value",color_continuous_scale="RdBu_r")
                        chart(fig)
                        st.caption("Features V1–V28 are anonymised components. SHAP contributions explain model output, not causal fraud drivers.")
                    except Exception as exc:
                        st.error(f"Explanation unavailable for this model/environment: {exc}")
if page == "Model Lab":
    st.subheader("Model evaluation lab")
    if report:
        st.markdown("**Validation model comparison**")
        rows=[]
        for name, m in report.get("validation_models",{}).items():
            rows.append({"Model":name,"PR-AUC":m.get("average_precision",0),
                         "ROC-AUC":m.get("roc_auc"),"Recall @ 0.5":m.get("recall",0)})
        if rows:
            comparison=pd.DataFrame(rows)
            st.dataframe(comparison,use_container_width=True,hide_index=True)
            chart(px.bar(comparison,x="Model",y="PR-AUC",title="Validation PR-AUC by model",
                         color_discrete_sequence=[COLORS["teal"]]))
        st.caption(f"Selection: {report.get('selected_model','unknown')}. Split: {report.get('split','not recorded')}. Threshold selected on validation data.")
    if scored is not None:
        threshold=st.slider("Explore a different screening threshold (uploaded sample only)",
                            min_value=0.0,max_value=1.0,value=float(min(1,max(0,art["threshold"]))),step=.005)
        whatif=scored.risk_score>=threshold
        a,b=st.columns(2)
        a.metric("Would be flagged",int(whatif.sum()))
        b.metric("What-if flag rate",f"{whatif.mean():.1%}")
        st.caption("This does not change the trained model or its saved decision threshold.")
        if "Class" in scored.columns and scored.Class.nunique()==2:
            precision,recall,_=precision_recall_curve(scored.Class,scored.risk_score)
            chart(px.line(x=recall,y=precision,title="Uploaded sample precision–recall curve",
                          labels={"x":"Recall","y":"Precision"}))
if page == "Simulator":
    st.subheader("Simulated transaction stream")
    st.caption("Replay of an uploaded historical dataset; NOT a live payment feed.")
    if scored is None:
        st.info("Upload transactions to start the simulator.")
    else:
        n=st.slider("Transactions to replay",min_value=1,max_value=min(200,len(scored)),
                    value=min(30,len(scored)))
        if st.button("▶ Replay transactions",type="primary"):
            progress=st.progress(0)
            counter=st.empty()
            recent=st.empty()
            stream=scored.head(n)
            for i in range(n):
                progress.progress((i+1)/n)
                counter.metric("Processed",f"{i+1} / {n}",delta=f"{int(stream.iloc[:i+1].flagged.sum())} flagged")
                recent.dataframe(stream.iloc[max(0,i-5):i+1],hide_index=True,use_container_width=True)
                time.sleep(.025)
            st.success("Historical replay complete. No transactions were sent to a bank.")
if page == "Monitoring":
    st.subheader("Data quality & monitoring")
    if scored is None:
        st.info("Upload a dataset to view local monitoring indicators.")
    else:
        st.caption("Local batch diagnostics only. No production telemetry or training-reference drift is claimed.")
        a,b,c=st.columns(3)
        a.metric("Missing feature values",int(df[FEATURES].isna().sum().sum()))
        b.metric("Duplicate input rows",int(df[FEATURES].duplicated().sum()))
        c.metric("Flagged transactions",int(scored.flagged.sum()))
        fig=px.scatter(scored,x="Time",y="risk_score",color="flagged",
                       color_discrete_map={True:COLORS["red"],False:COLORS["teal"]},
                       title="Risk scores over transaction time",hover_data=["Amount"])
        chart(fig)
        st.markdown("**Distribution comparison (first half vs second half of uploaded batch)**")
        if len(scored)>=20:
            mid=len(scored)//2
            h1=scored.iloc[:mid].risk_score
            h2=scored.iloc[mid:].risk_score
            st.write(f"First-half mean score: **{h1.mean():.4f}** · Second-half mean score: **{h2.mean():.4f}**")
            st.caption("Descriptive batch comparison, not a formal drift detector. Production drift requires reference distributions and alert thresholds.")
        else:
            st.info("At least 20 transactions are needed for a meaningful half-batch comparison.")
st.divider()
st.caption("FraudShield AI V4 · Research demonstration · No real banking connection · Human review required")
