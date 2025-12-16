import json
import ast

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Exposure Job Transition Recommender",
    layout="wide"
)

# -----------------------------
# LOAD DATA (cached)
# -----------------------------
@st.cache_data
def load_df():
    return pd.read_csv("Occupations_with_summaries_and_exposure.csv")


@st.cache_resource
def load_embeddings():
    return np.load("SBERT_embeddings_summaries.npy")


@st.cache_data
def load_transitions():
    return pd.read_csv("ALL_top10_transitions_with_skills.csv")


# -----------------------------
# HELPERS
# -----------------------------
def pick_column(df: pd.DataFrame, candidates, label: str):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a column for {label}. "
        f"Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def ensure_exposure_levels_and_jitter(df: pd.DataFrame, exposure_col: str):
    max_score = df[exposure_col].max()
    low_cutoff = max_score / 3.0
    high_cutoff = 2.0 * max_score / 3.0

    if "Exposure_Level_3cat" not in df.columns:
        def categorize(score):
            if score <= low_cutoff:
                return "Low exposure"
            elif score <= high_cutoff:
                return "Medium exposure"
            else:
                return "High exposure"

        df["Exposure_Level_3cat"] = df[exposure_col].apply(categorize)

    level_col = "Exposure_Level_3cat"
    df[level_col] = pd.Categorical(
        df[level_col],
        categories=["Low exposure", "Medium exposure", "High exposure"],
        ordered=True
    )

    if "jitter" not in df.columns:
        np.random.seed(42)
        df["jitter"] = np.random.uniform(-0.1, 0.1, size=len(df))

    return level_col, "jitter", low_cutoff, high_cutoff


def build_exposure_plot(
    df_plot: pd.DataFrame,
    exposure_col: str,
    level_col: str,
    jitter_col: str,
    occ_col: str,
    low_cutoff: float,
    high_cutoff: float,
):
    df_plot = df_plot.copy()
    df_plot[level_col] = pd.Categorical(
        df_plot[level_col],
        categories=["Low exposure", "Medium exposure", "High exposure"],
        ordered=True
    )

    fig = px.scatter(
        df_plot,
        x=exposure_col,
        y=jitter_col,
        color=level_col,
        category_orders={level_col: ["Low exposure", "Medium exposure", "High exposure"]},
        hover_data={
            occ_col: True,
            exposure_col: ':.3f',
            level_col: True,
            jitter_col: False,
        },
        labels={
            exposure_col: "AI exposure score",
            jitter_col: "",
            level_col: "Exposure level",
        },
        title="Distribution of occupations by AI exposure level",
    )

    fig.add_vline(x=low_cutoff, line_dash="dash", line_color="grey")
    fig.add_vline(x=high_cutoff, line_dash="dash", line_color="grey")

    fig.update_traces(marker=dict(size=7, opacity=0.9))
    fig.update_layout(
        template="simple_white",
        font=dict(size=14),
        title=dict(x=0.0, xanchor="left"),
        legend=dict(
            title="Exposure level",
            y=0.5,
            x=1.03,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=60, r=140, t=70, b=60),
    )
    fig.update_yaxes(showticklabels=False, title_text="")
    return fig


def recommend_lower_exposure_jobs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    selected_idx: int,
    exposure_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    current_exposure = df.loc[selected_idx, exposure_col]

    vec = embeddings[selected_idx].reshape(1, -1)
    sims = cosine_similarity(vec, embeddings)[0]

    recs = df.copy().reset_index(drop=False).rename(columns={"index": "row_idx"})
    recs["similarity"] = sims

    recs = recs[recs["row_idx"] != selected_idx]
    recs = recs[recs[exposure_col] < current_exposure]

    recs["exposure_diff"] = current_exposure - recs[exposure_col]
    recs = recs.sort_values("similarity", ascending=False).head(top_n)
    return recs


def parse_list_cell(x):
    """
    Handles list-like strings OR delimited strings (including your '|' delimiter).
    """
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []

    # Try JSON / python list
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj, list):
                return [str(v).strip() for v in obj if str(v).strip()]
        except Exception:
            pass

    # If it's pipe-delimited (your case), split by '|'
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return parts

    # Fallback: commas/newlines
    s = s.replace("\n", ",")
    parts = [p.strip(" -•\t") for p in s.split(",") if p.strip(" -•\t")]
    return parts


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("AI Exposure Job Transition Recommender")

    # Load
    try:
        df = load_df()
        embeddings = load_embeddings()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Detect columns
    try:
        occ_col = pick_column(df, ["Occupation", "Element Name"], "occupation name")
        exposure_col = pick_column(
            df,
            ["Final_ExposureScore", "Exposure_Score", "Exposure_Score_x", "Exposure_Score_y", "MEAN", "Exposure score"],
            "AI exposure score",
        )
        code_col = pick_column(df, ["O*NET-SOC Code", "ONET-SOC Code", "SOC Code"], "O*NET-SOC Code")
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Remove rows with missing exposure score
    mask_valid = df[exposure_col].notna()
    df = df.loc[mask_valid].copy()
    embeddings = embeddings[mask_valid.to_numpy()]
    df = df.reset_index(drop=True)

    # Exposure levels + jitter
    level_col, jitter_col, low_cutoff, high_cutoff = ensure_exposure_levels_and_jitter(df, exposure_col)

    # Sidebar (occupation names only)
    occupations = sorted(df[occ_col].dropna().unique().tolist())
    st.sidebar.header("Choose your occupation")
    occupation_choice = st.sidebar.selectbox(
        "Occupation",
        options=["Show all occupations"] + occupations,
        index=0,
    )

    if occupation_choice == "Show all occupations":
        selected_idx = None
    else:
        matches = df.index[df[occ_col] == occupation_choice].tolist()
        selected_idx = matches[0] if matches else None

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["AI exposure overview", "Recommended lower-exposure occupations", "Skill gap"]
    )

    # =========================
    # TAB 1
    # =========================
    with tab1:
        st.subheader("AI exposure overview")

        if selected_idx is None:
            plot_df = df
        else:
            plot_df = df.loc[[selected_idx]]

            st.markdown("### Your occupation")
            c1, c2, c3 = st.columns(3)
            c1.metric("Occupation", str(df.loc[selected_idx, occ_col]))
            c2.metric("AI exposure score", f"{df.loc[selected_idx, exposure_col]:.3f}")
            c3.metric("Exposure level", str(df.loc[selected_idx, level_col]))
            st.markdown("---")

        fig = build_exposure_plot(
            df_plot=plot_df,
            exposure_col=exposure_col,
            level_col=level_col,
            jitter_col=jitter_col,
            occ_col=occ_col,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 2
    # =========================
    with tab2:
        st.subheader("Recommended lower-exposure occupations")

        if selected_idx is None:
            st.info("Select your occupation in the sidebar to see recommendations.")
        else:
            st.markdown(
                "These occupations are **similar in content** to your current job "
                "but have a **lower AI exposure score**."
            )

            recs = recommend_lower_exposure_jobs(
                df=df,
                embeddings=embeddings,
                selected_idx=selected_idx,
                exposure_col=exposure_col,
                top_n=10,
            )

            if recs.empty:
                st.info("No lower-exposure occupations found for this job.")
            else:
                # Customer friendly: no SOC code in the table
                display_cols = [occ_col, exposure_col, "exposure_diff", "similarity"]
                display_cols = [c for c in display_cols if c in recs.columns]

                st.dataframe(
                    recs[display_cols].rename(
                        columns={
                            occ_col: "Occupation",
                            exposure_col: "AI exposure score",
                            "exposure_diff": "How much lower (Δ exposure)",
                            "similarity": "Similarity (0–1)",
                        }
                    ),
                    use_container_width=True,
                )

                st.caption(
                    "Similarity is based on SBERT embeddings of occupation summaries. "
                    "Only occupations with a lower AI exposure score than your current job are shown."
                )

                st.markdown("---")
                st.markdown("### Visual comparison with your current job")

                current_exposure = float(df.loc[selected_idx, exposure_col])

                vis_df = recs.head(10).copy()
                vis_df["Occupation_short"] = vis_df[occ_col].astype(str).str.slice(0, 45)

                fig_rec = px.scatter(
                    vis_df,
                    x=exposure_col,
                    y="Occupation_short",
                    size="similarity",
                    color="similarity",
                    hover_data={
                        occ_col: True,
                        exposure_col: ':.3f',
                        "similarity": ':.3f',
                        "exposure_diff": ':.3f',
                    },
                    labels={
                        exposure_col: "AI exposure score",
                        "Occupation_short": "Occupation",
                        "similarity": "Similarity (0–1)",
                    },
                    title="Recommended occupations vs. your current AI exposure score",
                )

                fig_rec.add_vline(x=current_exposure, line_dash="dash", line_color="grey")
                fig_rec.update_layout(
                    template="simple_white",
                    font=dict(size=13),
                    title=dict(x=0.05, xanchor="left"),
                    margin=dict(l=10, r=10, t=60, b=10),
                )

                st.plotly_chart(fig_rec, use_container_width=True)

                st.caption(
                    "Each dot is a recommended occupation. The vertical line shows your current job's AI exposure score. "
                    "Dots further to the left are safer (lower exposure), and larger / darker dots are more similar."
                )

    # =========================
    # TAB 3 (Customer friendly)
    # =========================
    with tab3:
        st.subheader("Skill gap")

        if selected_idx is None:
            st.info("Select your occupation in the sidebar first.")
            st.stop()

        trans = load_transitions()

        # Match current occupation using SOC in the transitions file (we won't display it)
        current_soc = str(df.loc[selected_idx, code_col])
        trans_current = trans[trans["source_soc"].astype(str) == current_soc].copy()

        if trans_current.empty:
            st.info("No skill transitions found for this occupation.")
            st.stop()

        st.markdown(
            "Pick one of the top recommended occupations to see which skills you already have "
            "and which skills you may need to gain."
        )

        # Dropdown shows ONLY occupation names (no codes)
        target_labels = [str(r["target_name"]) for _, r in trans_current.iterrows()]
        choice = st.selectbox("Choose a target occupation", target_labels)
        row = trans_current.iloc[target_labels.index(choice)]

        # Numbers
        similarity = float(row["similarity"])
        from_exp = float(row["source_ai_exposure"])
        to_exp = float(row["target_ai_exposure"])
        delta = float(row["delta_exposure"])

        shared_skills = parse_list_cell(row["shared_skills"])
        missing_skills = parse_list_cell(row["missing_skills"])[:5]  # ✅ top 5 only

        # Customer friendly summary (NO SOC + NO FROM/TO labels)
        st.markdown(
            f"""
**Current job:** {row['source_name']}  
**Target job:** {row['target_name']}  

**Similarity:** {similarity:.3f}  
**AI exposure change:** {from_exp:.3f} → {to_exp:.3f} (Δ={delta:.3f})
"""
        )

        st.markdown("---")

        # Two columns: shared vs missing
        left, right = st.columns(2)

        with left:
            st.markdown("### ✅ Shared skills")
            if shared_skills:
                for s in shared_skills:
                    st.markdown(f"- {s}")
            else:
                st.info("No shared skills listed for this transition.")

        with right:
            st.markdown("### ➕ Top missing skills")
            if missing_skills:
                for s in missing_skills:
                    st.markdown(f"- {s}")
            else:
                st.success("No missing skills listed for this transition.")


if __name__ == "__main__":
    main()



