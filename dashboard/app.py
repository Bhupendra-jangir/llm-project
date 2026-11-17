"""
dashboard/app.py
================
Multi-page Streamlit dashboard for the Enhanced LLM Evaluation Framework.

Pages:
  1. Overview — model ranking table (human vs metrics vs unified)
  2. Correlation heatmap + CIs
  3. Bias analysis plots
  4. Unified score weight explorer (live sliders)
  5. Single-sample inspector
  6. Human rating UI (Experiment D)
  7. Upload-your-own (live G-Eval)

The dashboard loads from results/ and runs without API keys.
Mock data files (results/_mock/) are clearly labeled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(
    page_title="LLM Evaluation Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── Custom CSS ─────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252d3e);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .label { font-size: 0.85rem; color: #718096; font-weight: 500; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #68d391; }

    .mock-banner {
        background: linear-gradient(135deg, #744210, #92400e);
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 10px 16px;
        color: #fde68a;
        font-weight: 600;
        margin-bottom: 16px;
    }

    .stTab [data-baseweb="tab"] {
        font-weight: 500;
    }

    .info-box {
        background: #1a365d;
        border-left: 4px solid #4299e1;
        padding: 12px 16px;
        border-radius: 4px;
        color: #bee3f8;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Helpers ─────────────────────────

RESULTS_DIR = ROOT / "results"
MOCK_DIR = ROOT / "results" / "_mock"
DATA_DIR = ROOT / "data"

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


@st.cache_data(ttl=300)
def load_results_df(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def mock_banner():
    st.markdown(
        '<div class="mock-banner">⚠️ MOCK DATA — These results were generated with the MockLLMClient. '
        'Add an API key and rerun experiments for real results.</div>',
        unsafe_allow_html=True,
    )


def _detect_results_dir() -> tuple[Path, bool]:
    """Return the best available results directory and whether it's mock."""
    # Prefer real results
    real_path = RESULTS_DIR / "correlations.csv"
    if real_path.exists():
        return RESULTS_DIR, False
    mock_path = MOCK_DIR / "correlations_mock.csv"
    if mock_path.exists():
        return MOCK_DIR, True
    return RESULTS_DIR, False


# ───────────────────────── Sidebar ─────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/null/artificial-intelligence.png", width=64)
st.sidebar.title("LLM Eval Framework")
st.sidebar.markdown("**Enhanced G-Eval + Multi-Metric**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate to",
    [
        "📊 Overview & Rankings",
        "🔥 Correlation Heatmap",
        "⚖️ Bias Analysis",
        "🎛️ Unified Score Explorer",
        "🔍 Sample Inspector",
        "👤 Human Rating UI",
        "🚀 Live G-Eval",
    ],
)

res_dir, is_mock = _detect_results_dir()

if is_mock:
    st.sidebar.warning("⚠️ Using mock data. Add API keys for real results.")

st.sidebar.divider()
st.sidebar.caption(f"Results dir: `{res_dir.name}/`")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: Overview & Rankings
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview & Rankings":
    st.title("📊 Overview & Model Rankings")
    if is_mock:
        mock_banner()

    st.markdown("""
    **Research question**: Can a multi-metric LLM evaluation framework combining G-Eval with
    traditional metrics and human-alignment analysis provide more reliable model rankings
    than any single automatic evaluation metric?
    """)

    # Load traditional metrics
    trad_df = load_results_df(str(res_dir / "traditional_metrics.csv"))
    if trad_df is None:
        trad_df = load_results_df(str(RESULTS_DIR / "traditional_metrics.csv"))

    if trad_df is not None:
        st.subheader("System Rankings by Metric")
        col1, col2 = st.columns([1, 2])

        with col1:
            human_dim = st.selectbox("Human dimension", DIMENSIONS + ["overall"])
            metric_to_rank = st.selectbox(
                "Metric to rank by",
                ["human_" + human_dim, "rouge1_ref", "rouge2_ref", "bertscore_F1_ref", "bleu_ref"],
            )

        human_col = f"human_{human_dim}"
        ranking_df = trad_df.groupby("system_id").agg(
            human_score=(human_col, "mean") if human_col in trad_df.columns else ("rouge1_ref", "mean"),
            rouge1=(  "rouge1_ref", "mean"),
            bertscore=("bertscore_F1_ref", "mean"),
            bleu=("bleu_ref", "mean"),
        ).reset_index()

        # Check for G-Eval scores
        geval_path = res_dir / "geval_scores_full_mock.csv" if is_mock else res_dir / "geval_scores_full.csv"
        if geval_path.exists():
            geval_df_raw = pd.read_csv(geval_path)
            overall_df = geval_df_raw[geval_df_raw["dimension"] == "overall"].groupby("system_id")["weighted_score"].mean()
            ranking_df = ranking_df.merge(
                overall_df.rename("geval_overall").reset_index(),
                on="system_id", how="left"
            )

        ranking_df = ranking_df.sort_values("human_score", ascending=False).reset_index(drop=True)
        ranking_df.index += 1

        with col2:
            st.dataframe(
                ranking_df.style.background_gradient(cmap="RdYlGn", subset=["human_score"])
                           .format({c: "{:.4f}" for c in ranking_df.select_dtypes(include=float).columns}),
                use_container_width=True,
            )

        # Summary stats
        st.subheader("Dataset Statistics")
        try:
            from src.utils import load_summeval, records_to_dataframe
            records = load_summeval()
            df_all = records_to_dataframe(records)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df_all):,}")
            c2.metric("Articles", f"{df_all['doc_id'].nunique()}")
            c3.metric("Systems", f"{df_all['system_id'].nunique()}")
            c4.metric("Human Dims", "5")
        except Exception:
            pass
    else:
        st.info("Run `python experiments/run_exp_b.py` to generate traditional metrics.")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════
elif page == "🔥 Correlation Heatmap":
    st.title("🔥 Correlation Heatmap")
    if is_mock:
        mock_banner()

    suffix = "_mock" if is_mock else ""
    corr_df = load_results_df(str(res_dir / f"correlations{suffix}.csv"))

    if corr_df is None:
        st.info("Run `python experiments/run_exp_e.py` to generate correlations.")
    else:
        corr_type = st.radio(
            "Correlation type",
            ["summary_spearman", "sample_spearman", "system_spearman"],
            horizontal=True,
        )

        pivot = corr_df.pivot_table(
            index="metric", columns="human_dim", values=corr_type
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.5 + 2)))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={"label": corr_type},
            linewidths=0.5,
        )
        ax.set_title(f"{corr_type.replace('_', ' ').title()}", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)
        plt.close()

        # Show top metrics with CIs
        st.subheader("Top Metrics with 95% Bootstrap CIs")
        if "summary_spearman_ci_lo" in corr_df.columns:
            top = corr_df[corr_df["human_dim"] == "overall"].sort_values(
                "summary_spearman", ascending=False
            ).head(10)
            st.dataframe(
                top[["metric", "summary_spearman", "summary_spearman_ci_lo",
                      "summary_spearman_ci_hi", "system_spearman"]].round(4),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════
# PAGE 3: Bias Analysis
# ═══════════════════════════════════════════════════════════════
elif page == "⚖️ Bias Analysis":
    st.title("⚖️ Evaluator Bias Analysis")
    if is_mock:
        mock_banner()

    suffix = "_mock" if is_mock else ""
    bias_path = res_dir / f"exp_f_bias_summary{suffix}.json"

    if not bias_path.exists():
        st.info("Run `python experiments/run_exp_f.py` to generate bias analysis.")
    else:
        with open(bias_path) as f:
            bias_data = json.load(f)

        if not bias_data:
            st.warning("No bias data found.")
        else:
            metric_names = [b["metric"] for b in bias_data]
            selected = st.selectbox("Select metric", metric_names)
            bias_entry = next(b for b in bias_data if b["metric"] == selected)

            st.subheader("1. Per-System Bias")
            sys_bias_df = pd.DataFrame(bias_entry.get("system_bias", []))
            if not sys_bias_df.empty:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12, 4))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#161b27")
                colors = ["#e74c3c" if b > 0 else "#3498db" for b in sys_bias_df["bias"]]
                ax.bar(sys_bias_df["system_id"], sys_bias_df["bias"], color=colors, alpha=0.85)
                ax.axhline(0, color="#718096", linewidth=1)
                ax.set_xlabel("System ID", color="white")
                ax.set_ylabel("Mean Z-score Bias", color="white")
                ax.tick_params(colors="white", axis="x", rotation=45)
                ax.tick_params(colors="white", axis="y")
                ax.set_title(f"Per-System Bias: {selected}", color="white", fontsize=14)
                st.pyplot(fig)
                plt.close()

            st.subheader("2. Length Bias")
            lb = bias_entry.get("length_bias", {})
            if lb:
                col1, col2 = st.columns(2)
                col1.metric("Metric ↔ Length (Spearman)", f"{lb.get('spearman_metric_length', 'N/A'):.4f}")
                col2.metric("Partial (controlling human)", f"{lb.get('partial_spearman_metric_length_given_human', 'N/A'):.4f}")

            st.subheader("3. System Type Bias")
            type_bias = pd.DataFrame(bias_entry.get("system_type_bias", []))
            if not type_bias.empty:
                st.dataframe(type_bias.round(4), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: Unified Score Weight Explorer
# ═══════════════════════════════════════════════════════════════
elif page == "🎛️ Unified Score Explorer":
    st.title("🎛️ Unified Score Weight Explorer")
    if is_mock:
        mock_banner()

    st.markdown("Adjust weights and see live correlation with human scores.")

    col1, col2 = st.columns([1, 2])
    with col1:
        w_geval = st.slider("G-Eval weight", 0.0, 1.0, 0.40, 0.05)
        w_bert = st.slider("BERTScore weight", 0.0, 1.0, 0.25, 0.05)
        w_rouge = st.slider("ROUGE weight", 0.0, 1.0, 0.15, 0.05)
        w_ens = st.slider("Ensemble agreement weight", 0.0, 1.0, 0.20, 0.05)
        total = w_geval + w_bert + w_rouge + w_ens

        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total:.2f} (should be 1.0). Will be normalized.")

    with col2:
        trad_df = load_results_df(str(res_dir / "traditional_metrics.csv"))
        if trad_df is None:
            trad_df = load_results_df(str(RESULTS_DIR / "traditional_metrics.csv"))

        if trad_df is not None:
            try:
                from src.unified_score import compute_unified_score
                from src.utils import summary_level_correlation

                weights = {
                    "geval": w_geval / total if total > 0 else 0.25,
                    "bertscore": w_bert / total if total > 0 else 0.25,
                    "rouge": w_rouge / total if total > 0 else 0.25,
                    "ensemble_agreement": w_ens / total if total > 0 else 0.25,
                }
                scores = compute_unified_score(trad_df, weights=weights)
                human = trad_df["human_overall"].values.astype(float) if "human_overall" in trad_df.columns else np.zeros(len(trad_df))
                doc_ids = trad_df["doc_id"].values

                valid = ~(np.isnan(scores) | np.isnan(human))
                corr = summary_level_correlation(scores[valid], human[valid], doc_ids[valid])

                st.metric("Summary-Level Spearman ρ", f"{corr['spearman']:.4f}")
                st.metric("Summary-Level Kendall τ", f"{corr['kendall']:.4f}")
                st.caption("Based on SummEval human overall scores.")
            except Exception as e:
                st.error(f"Error computing unified score: {e}")
        else:
            st.info("Run `python experiments/run_exp_b.py` first.")

    # Show best grid weights if available
    suffix = "_mock" if is_mock else ""
    grid_path = res_dir / f"exp_g_grid_search{suffix}.csv"
    if grid_path.exists():
        st.subheader("Best Weights from Grid Search")
        grid_df = pd.read_csv(grid_path).head(5)
        st.dataframe(grid_df.round(4), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5: Sample Inspector
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Sample Inspector":
    st.title("🔍 Single Sample Inspector")

    try:
        from src.utils import load_summeval
        records = load_summeval()

        col1, col2 = st.columns([1, 2])
        with col1:
            doc_ids = sorted(list(set(r["doc_id"] for r in records)))
            selected_doc = st.selectbox("Select document", doc_ids)
            doc_records = [r for r in records if r["doc_id"] == selected_doc]
            system_ids = [r["system_id"] for r in doc_records]
            selected_sys = st.selectbox("Select system", system_ids)

        record = next(r for r in doc_records if r["system_id"] == selected_sys)

        with col2:
            st.subheader("Source Article")
            st.text_area("Source", record["source"][:1000] + "...", height=150, disabled=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Reference Summary")
            st.text_area("Reference", record["reference"], height=100, disabled=True)
        with col4:
            st.subheader("System Output")
            st.text_area("System Output", record["system_output"], height=100, disabled=True)

        st.subheader("Human Scores (Ground Truth)")
        scores = record["scores"]
        cols = st.columns(5)
        for i, (dim, val) in enumerate(scores.items()):
            cols[i].metric(dim.capitalize(), f"{val:.3f}")

        # Show metric scores if available
        trad_df = load_results_df(str(res_dir / "traditional_metrics.csv"))
        if trad_df is not None:
            matching = trad_df[
                (trad_df["doc_id"] == record["doc_id"]) &
                (trad_df["system_id"] == record["system_id"])
            ]
            if not matching.empty:
                st.subheader("Automatic Metric Scores")
                row = matching.iloc[0]
                metric_cols = ["rouge1_ref", "rouge2_ref", "rougeL_ref", "bleu_ref",
                               "bertscore_F1_ref", "bertscore_F1_src"]
                disp = {c: round(row[c], 4) for c in metric_cols if c in row.index}
                st.json(disp)

    except FileNotFoundError:
        st.info("SummEval data not found. Ensure data/summeval.json exists.")


# ═══════════════════════════════════════════════════════════════
# PAGE 6: Human Rating UI
# ═══════════════════════════════════════════════════════════════
elif page == "👤 Human Rating UI":
    st.title("👤 Human Rating Interface")
    st.markdown("""
    Rate summaries on four quality dimensions to contribute to Experiment D.
    Your ratings will be saved to `data/human_ratings.csv`.
    """)

    template_path = ROOT / "dashboard" / "human_rating_template.csv"
    ratings_path = DATA_DIR / "human_ratings.csv"

    if not template_path.exists():
        st.info("Rating template not generated yet. Run `python experiments/run_exp_d.py` first.")
        st.stop()

    template_df = pd.read_csv(template_path)
    # Load existing ratings
    if ratings_path.exists():
        existing_ratings = pd.read_csv(ratings_path)
        rated_ids = set(existing_ratings["id"].tolist())
    else:
        existing_ratings = pd.DataFrame()
        rated_ids = set()

    unrated = template_df[~template_df["id"].isin(rated_ids)]
    st.info(f"📋 {len(template_df)} total samples | {len(rated_ids)} rated | {len(unrated)} remaining")

    if len(unrated) == 0:
        st.success("✅ All samples rated! Check `data/human_ratings.csv`.")
        st.dataframe(existing_ratings.tail(10))
    else:
        sample = unrated.iloc[0]

        st.subheader(f"Sample #{int(sample['id'])} | System: {sample['system_id']}")
        st.text_area("Source Article", sample["source"][:800] + "...", height=120, disabled=True)
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Reference", sample["reference"], height=80, disabled=True)
        with col2:
            st.text_area("Summary to Rate", sample["system_output"], height=80, disabled=True)

        st.markdown("### Your Ratings")
        dim_descriptions = {
            "coherence": "Well-structured and logically organized? (1=poor, 5=excellent)",
            "consistency": "Factually aligned with the source? (1=many errors, 5=fully consistent)",
            "fluency": "Grammar, spelling, readability? (1=poor, 2=fair, 3=good)",
            "relevance": "Covers key information without redundancy? (1=poor, 5=excellent)",
        }
        dim_ranges = {"coherence": (1, 5), "consistency": (1, 5), "fluency": (1, 3), "relevance": (1, 5)}

        ratings_input = {}
        for dim in DIMENSIONS:
            min_v, max_v = dim_ranges[dim]
            ratings_input[dim] = st.slider(
                f"{dim.capitalize()}: {dim_descriptions[dim]}",
                min_value=min_v, max_value=max_v, value=(min_v + max_v) // 2,
                key=f"rate_{dim}_{sample['id']}",
            )

        if st.button("✅ Submit Rating", type="primary"):
            new_row = {
                "id": sample["id"],
                "doc_id": sample["doc_id"],
                "system_id": sample["system_id"],
                "rated_coherence": ratings_input["coherence"],
                "rated_consistency": ratings_input["consistency"],
                "rated_fluency": ratings_input["fluency"],
                "rated_relevance": ratings_input["relevance"],
            }
            existing_ratings = pd.concat(
                [existing_ratings, pd.DataFrame([new_row])], ignore_index=True
            )
            existing_ratings.to_csv(ratings_path, index=False)
            st.success(f"✅ Rating saved! ({len(existing_ratings)}/{len(template_df)} complete)")
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 7: Live G-Eval
# ═══════════════════════════════════════════════════════════════
elif page == "🚀 Live G-Eval":
    st.title("🚀 Live G-Eval Scorer")
    st.markdown("Paste an article and summary to get G-Eval scores in real time.")

    has_key = False
    env_path = ROOT / ".env"
    if env_path.exists():
        content = env_path.read_text()
        has_key = any(k in content and "your_" not in content
                      for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"])

    if not has_key:
        st.warning(
            "⚠️ No API key detected. Using MockLLMClient (scores are random, not meaningful). "
            "Add a key to `.env` for real evaluation."
        )
        use_mock = True
    else:
        use_mock = st.checkbox("Use mock client (fast, random scores)", value=False)

    article = st.text_area("Source Article", height=200, placeholder="Paste the news article here...")
    summary = st.text_area("Summary to Evaluate", height=100, placeholder="Paste the summary here...")

    if st.button("🔍 Evaluate", type="primary"):
        if not article or not summary:
            st.error("Please provide both article and summary.")
        else:
            with st.spinner("Evaluating..."):
                try:
                    import asyncio
                    from src.llm_clients import create_client, DiskCache
                    from src.geval_evaluator import GEvalEvaluator
                    from src.utils import load_config

                    cfg = load_config()
                    if use_mock:
                        provider, model = "mock", "mock-model"
                    else:
                        provider = cfg["llm"]["provider"]
                        model = cfg["llm"]["model"]

                    client = create_client(
                        provider=provider, model=model,
                        cache=DiskCache(ROOT / "data" / "cache" / "dashboard_cache.db"),
                    )
                    evaluator = GEvalEvaluator(client=client)

                    async def evaluate():
                        await evaluator.setup_cot()
                        results = {}
                        for dim in DIMENSIONS:
                            r = await evaluator.evaluate_record(article, summary, dim)
                            results[dim] = r
                        return results

                    results = asyncio.run(evaluate())

                    st.subheader("Results")
                    cols = st.columns(4)
                    for i, dim in enumerate(DIMENSIONS):
                        r = results[dim]
                        ws = r.get("weighted_score")
                        score_str = f"{ws:.3f}" if ws is not None and not (isinstance(ws, float) and ws != ws) else "N/A"
                        cols[i].metric(
                            dim.capitalize(),
                            score_str,
                            help=f"Raw response: {r.get('raw_text', 'N/A')[:50]}"
                        )

                    if use_mock:
                        st.caption("⚠️ MOCK scores — not meaningful. Add API key for real evaluation.")

                    # Show probability distribution
                    for dim in DIMENSIONS:
                        weights = results[dim].get("logprob_weights", {})
                        if weights:
                            st.subheader(f"{dim.capitalize()} Score Distribution")
                            chart_data = pd.DataFrame(
                                {"Score": list(weights.keys()), "Probability": list(weights.values())}
                            )
                            st.bar_chart(chart_data.set_index("Score"))

                except Exception as e:
                    st.error(f"Evaluation error: {e}")
                    st.exception(e)
