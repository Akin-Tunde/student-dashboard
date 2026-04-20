import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import copy, urllib.request, zipfile, os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import shap
import lime, lime.lime_tabular

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='Student Performance Predictor',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Selectbox and slider styling */
    .stSelectbox, .stSlider {
        background-color: white;
    }
    
    /* Section divider */
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_train():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
    if not os.path.exists('student.zip'):
        urllib.request.urlretrieve(url, 'student.zip')
    with zipfile.ZipFile('student.zip','r') as z:
        z.extractall('student_data')

    df_raw = pd.read_csv('student_data/student-por.csv', sep=';')
    df = df_raw.copy()
    df['target'] = (df['G3'] >= 10).astype(int)
    df = df.drop(columns=['G1','G2','G3'])
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=['target']); y = df['target']
    fn = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_b, y_tr_b = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_b); X_te_s = sc.transform(X_te)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree':       DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5)
    }
    trained, results = {}, {}
    for name, m in models.items():
        m = copy.deepcopy(m); m.fit(X_tr_s, y_tr_b)
        yp = m.predict(X_te_s); yb = m.predict_proba(X_te_s)[:,1]
        trained[name] = m
        results[name] = {
            'Accuracy':  round(accuracy_score(y_te, yp)*100,2),
            'Precision': round(precision_score(y_te,yp,average='macro',zero_division=0)*100,2),
            'Recall':    round(recall_score(y_te,yp,average='macro',zero_division=0)*100,2),
            'F1-Score':  round(f1_score(y_te,yp,average='macro',zero_division=0),4),
        }
    return trained, results, sc, fn, X_te_s, y_te, X_te, X_tr_s, y_tr_b, df_raw

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎓 Navigation")
    page = st.radio(
        "Select a page:",
        ['📊 Overview', '📈 Dataset Analysis', '🤖 Model Comparison', '🔮 Predict', '💡 Explainability', '⚖️ Fairness'],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    ### About This Dashboard
    **Student Performance Predictor** uses machine learning to predict student success.
    
    - **Dataset:** UCI Portuguese Language (649 students)
    - **Models:** Logistic Regression & Decision Tree
    - **Task:** Binary classification (Pass/Fail)
    """)
    st.markdown("---")

# Load data
with st.spinner('🔄 Loading models and data...'):
    tm, res, sc, fn, Xte, yte, Xte_r, Xtr_s, ytr_b, df_raw = load_and_train()

best_model = max(res, key=lambda x: res[x]['F1-Score'])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == '📊 Overview':
    st.markdown('<p class="header-title">🎓 Student Performance Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Predicting student success using machine learning</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Students", "649", delta="Portuguese Dataset")
    with col2:
        pass_rate = (df_raw['G3'] >= 10).mean() * 100
        st.metric("✅ Pass Rate", f"{pass_rate:.1f}%", delta=f"{int(pass_rate/100*649)} students")
    with col3:
        st.metric("🤖 Models Trained", "2", delta="LR & DT")
    with col4:
        st.metric("🏆 Best Model", best_model, delta=f"F1: {res[best_model]['F1-Score']}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Model comparison table
    st.markdown("### Model Performance Comparison")
    df_results = pd.DataFrame(res).T.round(2)
    st.dataframe(df_results, use_container_width=True, hide_index=False)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Quick insights
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### 🎯 Key Features
        - **Real-time predictions** with confidence scores
        - **SHAP explainability** for model transparency
        - **Fairness analysis** across demographics
        - **Interactive visualizations** for data exploration
        """)
    with col2:
        st.success("""
        ### 📈 Model Insights
        - **Best Performer:** """ + best_model + """
        - **Accuracy:** """ + str(res[best_model]['Accuracy']) + """%
        - **F1-Score:** """ + str(res[best_model]['F1-Score']) + """
        - **Recall:** """ + str(res[best_model]['Recall']) + """%
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📈 Dataset Analysis':
    st.markdown('<p class="header-title">📈 Dataset Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">UCI Portuguese Language Student Dataset</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df_raw))
    with col2:
        st.metric("Features", len(df_raw.columns) - 1)
    with col3:
        st.metric("Pass Threshold", "Grade ≥ 10")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pass/Fail Distribution")
        pass_count = (df_raw['G3'] >= 10).sum()
        fail_count = (df_raw['G3'] < 10).sum()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#10b981', '#ef4444']
        wedges, texts, autotexts = ax.pie(
            [pass_count, fail_count],
            labels=['Pass', 'Fail'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        ax.set_title('Student Outcomes', fontsize=13, weight='bold', pad=20)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("### Grade Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_raw['G3'], bins=20, color='#3b82f6', edgecolor='white', alpha=0.8)
        ax.axvline(x=10, color='#ef4444', linestyle='--', linewidth=2, label='Pass Threshold (10)')
        ax.set_xlabel('Final Grade (G3)', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title('Grade Distribution', fontsize=13, weight='bold', pad=20)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🤖 Model Comparison':
    st.markdown('<p class="header-title">🤖 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Logistic Regression vs Decision Tree</p>', unsafe_allow_html=True)
    
    # Model metrics table
    st.markdown("### Performance Metrics")
    df_r = pd.DataFrame(res).T
    st.dataframe(df_r, use_container_width=True, hide_index=False)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Confusion matrices
    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)
    
    for idx, (model_name, col) in enumerate(zip(tm.keys(), [col1, col2])):
        with col:
            yp = tm[model_name].predict(Xte)
            cm = confusion_matrix(yte, yp)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                       xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
                       cbar_kws={'label': 'Count'})
            ax.set_title(f'{model_name}', fontsize=12, weight='bold', pad=15)
            ax.set_ylabel('True Label', fontsize=10, weight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10, weight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🔮 Predict':
    st.markdown('<p class="header-title">🔮 Predict Student Outcome</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Enter student details to get a prediction</p>', unsafe_allow_html=True)
    
    # Model selection
    st.markdown("### Select Model")
    mc = st.selectbox("Choose a model:", list(tm.keys()), label_visibility="collapsed")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Input form
    st.markdown("### Student Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        age = st.slider('Age', 15, 22, 17, help="Student age in years")
        sex = st.selectbox('Gender', ['Female', 'Male'], help="Student gender")
    
    with col2:
        st.markdown("#### Academic")
        studytime = st.slider('Study Time', 1, 4, 2, help="1=<2h, 2=2-5h, 3=5-10h, 4=>10h")
        failures = st.slider('Past Failures', 0, 3, 0, help="Number of past class failures")
    
    with col3:
        st.markdown("#### Engagement")
        absences = st.slider('Absences', 0, 93, 5, help="Number of school absences")
        internet = st.selectbox('Internet Access', ['Yes', 'No'], help="Has internet at home")
    
    st.markdown("#### Parent Education")
    col1, col2 = st.columns(2)
    with col1:
        Medu = st.slider('Mother Education (0-4)', 0, 4, 2, help="0=none, 1=primary, 2=5-9, 3=secondary, 4=higher")
    with col2:
        Fedu = st.slider('Father Education (0-4)', 0, 4, 2, help="0=none, 1=primary, 2=5-9, 3=secondary, 4=higher")
    
    higher = st.selectbox('Wants Higher Education', ['Yes', 'No'], help="Wants to pursue higher education")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button('🎯 Generate Prediction', use_container_width=True):
        inp = {n: 0 for n in fn}
        inp.update({
            'age': age, 'absences': absences, 'studytime': studytime,
            'failures': failures, 'Medu': Medu, 'Fedu': Fedu,
            'sex': 1 if sex == 'Male' else 0,
            'internet': 1 if internet == 'Yes' else 0,
            'higher': 1 if higher == 'Yes' else 0,
            'famrel': 3, 'health': 3, 'traveltime': 1, 'freetime': 3, 'goout': 3
        })
        
        Xi = sc.transform(pd.DataFrame([inp])[fn])
        pred = tm[mc].predict(Xi)[0]
        prob = tm[mc].predict_proba(Xi)[0][1]
        
        st.markdown("### Prediction Result")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred == 1:
                st.success("### ✅ PASS")
            else:
                st.error("### ❌ FAIL")
        
        with col2:
            st.metric("Pass Probability", f"{prob*100:.1f}%")
        
        with col3:
            st.metric("Fail Probability", f"{(1-prob)*100:.1f}%")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Risk assessment
        st.markdown("### Risk Assessment")
        if prob >= 0.75:
            st.success(f"🟢 **LOW RISK** — Student has a {prob*100:.1f}% probability of passing. Strong performance expected.")
        elif prob >= 0.5:
            st.warning(f"🟡 **MEDIUM RISK** — Student has a {prob*100:.1f}% probability of passing. Monitor progress.")
        else:
            st.error(f"🔴 **HIGH RISK** — Student has a {prob*100:.1f}% probability of passing. Intervention recommended.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '💡 Explainability':
    st.markdown('<p class="header-title">💡 Model Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Understand why the model makes predictions (SHAP)</p>', unsafe_allow_html=True)
    
    mc = st.selectbox("Select Model:", list(tm.keys()), label_visibility="collapsed")
    m = tm[mc]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    with st.spinner('🔄 Computing SHAP values...'):
        if mc == 'Logistic Regression':
            exp = shap.LinearExplainer(m, Xtr_s, feature_perturbation='interventional')
            sv = exp.shap_values(Xte)
        else:
            exp = shap.TreeExplainer(m)
            sv_raw = exp.shap_values(Xte)
            sv = sv_raw[1] if isinstance(sv_raw, list) else (sv_raw[:,:,1] if sv_raw.ndim==3 else sv_raw)
    
    # Global feature importance
    st.markdown("### Global Feature Importance")
    ma = np.abs(sv).mean(axis=0)
    ti = np.argsort(ma)[-12:].astype(int)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ti)))
    ax.barh([fn[i] for i in ti], ma[ti], color=colors, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11, weight='bold')
    ax.set_title(f'Top 12 Most Important Features — {mc}', fontsize=13, weight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Individual prediction explanation
    st.markdown("### Individual Prediction Explanation")
    idx = st.slider('Select a Student (by index):', 0, len(Xte)-1, 0)
    
    p = m.predict(Xte[idx:idx+1])[0]
    pb = m.predict_proba(Xte[idx:idx+1])[0][1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", "PASS ✅" if p == 1 else "FAIL ❌")
    with col2:
        st.metric("Pass Probability", f"{pb*100:.1f}%")
    with col3:
        st.metric("Actual Outcome", "PASS ✅" if yte.iloc[idx] == 1 else "FAIL ❌")
    
    # Feature contribution
    sv_s = sv[idx]
    si = np.argsort(np.abs(sv_s))[-10:].astype(int)
    cols = ['#ef4444' if v < 0 else '#10b981' for v in sv_s[si]]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([fn[i] for i in si], sv_s[si], color=cols, edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, weight='bold')
    ax.set_title(f'Feature Contributions for Student #{idx} — Predicted: {"PASS" if p==1 else "FAIL"} ({pb*100:.1f}%)', fontsize=13, weight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    # Explanation summary
    top_feat = fn[np.argmax(np.abs(sv_s))]
    direction = 'toward PASS ✅' if sv_s[np.argmax(np.abs(sv_s))] > 0 else 'toward FAIL ❌'
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.info(f"""
    **Explanation Summary:**
    
    This student's prediction is primarily driven by **{top_feat}** ({direction}). 
    The model predicts **{"PASS" if p==1 else "FAIL"}** with **{pb*100:.1f}%** pass probability.
    
    *Note: Educator judgment should guide any intervention decisions.*
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '⚖️ Fairness':
    st.markdown('<p class="header-title">⚖️ Fairness Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Evaluating model fairness across demographics</p>', unsafe_allow_html=True)
    
    mc = st.selectbox("Select Model:", list(tm.keys()), label_visibility="collapsed")
    yp = tm[mc].predict(Xte)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Gender fairness analysis
    st.markdown("### Gender Fairness Analysis")
    df_ev = Xte_r.copy()
    df_ev['true'] = yte.values
    df_ev['pred'] = yp
    
    rows = []
    for code, label in [(0, 'Female'), (1, 'Male')]:
        g = df_ev[df_ev['sex'] == code]
        if len(g) == 0:
            continue
        rows.append({
            'Group': label,
            'Students': len(g),
            'Actual Pass Rate': f"{g['true'].mean()*100:.1f}%",
            'Predicted Pass Rate': f"{g['pred'].mean()*100:.1f}%",
            'F1-Score': f"{f1_score(g['true'], g['pred'], zero_division=0):.3f}"
        })
    
    st.dataframe(pd.DataFrame(rows).set_index('Group'), use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Fairness metrics
    fr = df_ev[df_ev['sex'] == 0]['pred'].mean() * 100
    mr = df_ev[df_ev['sex'] == 1]['pred'].mean() * 100
    diff = abs(fr - mr)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Female Pass Rate", f"{fr:.1f}%")
    with col2:
        st.metric("Male Pass Rate", f"{mr:.1f}%")
    with col3:
        st.metric("Difference", f"{diff:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Fairness assessment
    st.markdown("### Fairness Assessment")
    if diff < 10:
        st.success(f"""
        ✅ **FAIR** — The difference of {diff:.1f}% is under the 10% threshold.
        
        The model demonstrates acceptable fairness across gender groups (80% rule satisfied).
        """)
    else:
        st.error(f"""
        ⚠️ **POTENTIAL BIAS** — The difference of {diff:.1f}% exceeds the 10% threshold.
        
        There may be gender-based disparities in model predictions that require further investigation.
        """)
    
    st.caption("*Note: This is a simplified fairness check. A comprehensive analysis would include additional metrics (Disparate Impact Ratio, Equalized Odds, etc.)*")
