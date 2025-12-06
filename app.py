import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# Small helper: NumPy softmax (no SciPy needed)
# ---------------------------------------------------------
def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Numerically stable softmax using NumPy.
    Equivalent to scipy.special.softmax but avoids extra dependency.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_reviews(path: str) -> pd.DataFrame:
    """
    Load the Stage 1 real reviews dataset and apply required normalization.

    - Reads parquet file from `path`
    - Normalizes column names to lowercase and strips whitespace
    - Ensures publishedatdate is timezone-naive datetime
    - Ensures sentiment_from_stars exists:
        positive if stars >= 4, else negative
    """
    df = pd.read_parquet("data/stage1_business_full_clean.parquet")

    # Normalize column names to lowercase / strip spaces
    df.columns = [c.strip().lower() for c in df.columns]

    # Make sure publishedatdate is timezone-naive datetime
    if "publishedatdate" in df.columns:
        df["publishedatdate"] = pd.to_datetime(df["publishedatdate"], errors="coerce")

        # If timezone-aware, drop timezone info
        if pd.api.types.is_datetime64tz_dtype(df["publishedatdate"]):
            df["publishedatdate"] = df["publishedatdate"].dt.tz_convert(None)

    # Ensure sentiment_from_stars exists
    if "sentiment_from_stars" not in df.columns:
        if "stars" not in df.columns:
            raise ValueError("Column 'stars' not found; cannot derive sentiment_from_stars.")
        df["sentiment_from_stars"] = np.where(df["stars"] >= 4, "positive", "negative")

    # Ensure review_length_tokens exists (defensive)
    if "review_length_tokens" not in df.columns and "review_text" in df.columns:
        # Fallback: approximate by whitespace split
        df["review_length_tokens"] = df["review_text"].fillna("").str.split().str.len()

    return df


# ---------------------------------------------------------
# 2. SIDEBAR FILTERS
# ---------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sidebar widgets and return the filtered DataFrame.

    Filters:
    - Date range (publishedatdate)
    - Star rating range
    - Reviewer type (all / local guides / non-local)
    - Year (all + review_year)
    - Month (all + review_month_name)
    """
    st.sidebar.header("Filters")

    df_filtered = df.copy()

    # --- Date range filter ---
    if "publishedatdate" in df_filtered.columns:
        min_date = df_filtered["publishedatdate"].min()
        max_date = df_filtered["publishedatdate"].max()

        if pd.isna(min_date) or pd.isna(max_date):
            date_range = None
        else:
            # date_input expects date objects
            default_start = min_date.date()
            default_end = max_date.date()
            date_range = st.sidebar.date_input(
                "Review date range",
                value=(default_start, default_end),
                min_value=default_start,
                max_value=default_end,
            )

        if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            # Filter using .dt.date to avoid time boundary issues
            df_filtered = df_filtered[
                (df_filtered["publishedatdate"].dt.date >= start_date)
                & (df_filtered["publishedatdate"].dt.date <= end_date)
            ]

    # Handle complete filter-out early
    if df_filtered.empty:
        return df_filtered

    # --- Star rating range ---
    if "stars" in df_filtered.columns:
        min_star = int(df_filtered["stars"].min())
        max_star = int(df_filtered["stars"].max())
        star_min, star_max = st.sidebar.slider(
            "Star rating range",
            min_value=min_star,
            max_value=max_star,
            value=(min_star, max_star),
            step=1,
        )
        df_filtered = df_filtered[
            (df_filtered["stars"] >= star_min) & (df_filtered["stars"] <= star_max)
        ]

    if df_filtered.empty:
        return df_filtered

    # --- Reviewer type filter ---
    reviewer_type = st.sidebar.selectbox(
        "Reviewer type",
        options=["All", "Local Guides only", "Non-local guides only"],
    )

    if "islocalguide" in df_filtered.columns:
        if reviewer_type == "Local Guides only":
            df_filtered = df_filtered[df_filtered["islocalguide"] == True]
        elif reviewer_type == "Non-local guides only":
            df_filtered = df_filtered[df_filtered["islocalguide"] == False]

    if df_filtered.empty:
        return df_filtered

    # --- Year filter ---
    year_options = ["All"]
    if "review_year" in df_filtered.columns:
        years = sorted(df_filtered["review_year"].dropna().unique())
        year_options += [int(y) for y in years]

    selected_year = st.sidebar.selectbox("Year", options=year_options)

    if selected_year != "All" and "review_year" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["review_year"] == selected_year]

    if df_filtered.empty:
        return df_filtered

    # --- Month filter ---
    month_options = ["All"]
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    if "review_month_name" in df_filtered.columns:
        # Preserve calendar order
        available_months = sorted(
            df_filtered["review_month_name"].dropna().unique(),
            key=lambda m: month_order.index(m) if m in month_order else 999,
        )
        month_options += available_months

    selected_month = st.sidebar.selectbox("Month", options=month_options)

    if selected_month != "All" and "review_month_name" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["review_month_name"] == selected_month]

    return df_filtered


# ---------------------------------------------------------
# 3. KPI SECTION
# ---------------------------------------------------------
def show_kpis(df_filtered: pd.DataFrame) -> None:
    """Display top-level KPIs for the currently filtered dataset."""
    st.subheader("Key Metrics (Filtered View)")

    col1, col2, col3, col4 = st.columns(4)

    # 1. Total reviews
    with col1:
        if df_filtered.empty:
            st.metric("Total Reviews", "N/A")
        else:
            st.metric("Total Reviews", f"{len(df_filtered):,}")

    # 2. % Positive (based on sentiment_from_stars)
    with col2:
        if df_filtered.empty or "sentiment_from_stars" not in df_filtered.columns:
            st.metric("% Positive (Stars)", "N/A")
        else:
            pos_rate = (
                (df_filtered["sentiment_from_stars"] == "positive").mean() * 100.0
            )
            st.metric("% Positive (Stars)", f"{pos_rate:.1f}%")

    # 3. Average rating
    with col3:
        if df_filtered.empty or "stars" not in df_filtered.columns:
            st.metric("Average Rating", "N/A")
        else:
            avg_rating = df_filtered["stars"].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} / 5")

    # 4. Average review length (tokens)
    with col4:
        if df_filtered.empty or "review_length_tokens" not in df_filtered.columns:
            st.metric("Avg Length (Tokens)", "N/A")
        else:
            avg_len = df_filtered["review_length_tokens"].mean()
            st.metric("Avg Length (Tokens)", f"{avg_len:.1f}")

    st.caption(
        "All metrics and charts use the real Stage 1 dataset "
        "(no synthetic or augmented reviews) to preserve true customer behavior."
    )


# ---------------------------------------------------------
# 4. CHARTS & TABLES
# ---------------------------------------------------------
def show_sentiment_over_time(df_filtered: pd.DataFrame) -> None:
    """
    Sentiment over time: proportion of negative reviews per month.
    Business meaning: highlights when negative sentiment spikes
    so management can correlate with promotions, staffing, price changes, etc.
    """
    st.subheader("Sentiment Over Time (Based on Star Ratings)")

    if df_filtered.empty or "publishedatdate" not in df_filtered.columns:
        st.info("No data available for this view.")
        return

    df = df_filtered.dropna(subset=["publishedatdate", "sentiment_from_stars"]).copy()
    if df.empty:
        st.info("No data available after removing missing dates/labels.")
        return

    df["month"] = df["publishedatdate"].dt.to_period("M").dt.to_timestamp()
    df["is_negative"] = df["sentiment_from_stars"] == "negative"

    monthly = (
        df.groupby("month", as_index=False)["is_negative"]
        .mean()
        .rename(columns={"is_negative": "neg_rate"})
    )

    if monthly.empty:
        st.info("No data available for monthly sentiment aggregation.")
        return

    chart = (
        alt.Chart(monthly)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("neg_rate:Q", title="Proportion Negative", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                alt.Tooltip("neg_rate:Q", title="Proportion Negative", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)


def show_star_distribution(df_filtered: pd.DataFrame) -> None:
    """Bar chart of star rating distribution."""
    st.subheader("Star Rating Distribution")

    if df_filtered.empty or "stars" not in df_filtered.columns:
        st.info("No data available for this view.")
        return

    agg = (
        df_filtered.groupby("stars", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("stars")
    )

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("stars:O", title="Stars"),
            y=alt.Y("count:Q", title="Number of Reviews"),
            tooltip=[
                alt.Tooltip("stars:O", title="Stars"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def show_review_length_distribution(df_filtered: pd.DataFrame) -> None:
    """
    Histogram of review_length_tokens.
    Business meaning: longer reviews often contain detailed complaints or strong praise.
    This shows how deeply guests typically engage.
    """
    st.subheader("Review Length Distribution (Tokens)")

    if df_filtered.empty or "review_length_tokens" not in df_filtered.columns:
        st.info("No data available for this view.")
        return

    df = df_filtered.dropna(subset=["review_length_tokens"])
    if df.empty:
        st.info("No data available after removing missing review lengths.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "review_length_tokens:Q",
                bin=alt.Bin(maxbins=30),
                title="Review Length (tokens)",
            ),
            y=alt.Y("count():Q", title="Number of Reviews"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def show_recent_negative_reviews(df_filtered: pd.DataFrame) -> None:
    """
    Show the 20 most recent negative reviews.
    Acts as a 'high-risk review inbox' for managers.
    """
    st.subheader("Recent Negative Reviews (Top Complaints)")

    if df_filtered.empty or "sentiment_from_stars" not in df_filtered.columns:
        st.info("No data available for this view.")
        return

    df_neg = df_filtered[df_filtered["sentiment_from_stars"] == "negative"].copy()
    if df_neg.empty:
        st.info("No negative reviews for this filter selection.")
        return

    if "publishedatdate" in df_neg.columns:
        df_neg = df_neg.sort_values("publishedatdate", ascending=False)
    else:
        df_neg = df_neg.sort_index(ascending=False)

    cols_to_show = []
    for col in ["publishedatdate", "stars", "likescount", "review_length_tokens", "review_text"]:
        if col in df_neg.columns:
            cols_to_show.append(col)

    st.dataframe(df_neg[cols_to_show].head(20), use_container_width=True)


def show_most_liked_reviews(df_filtered: pd.DataFrame) -> None:
    """
    Show the 20 most liked reviews, highlighting 'hero stories' that marketing can reuse.
    """
    st.subheader("Most Liked Reviews")

    if df_filtered.empty or "likescount" not in df_filtered.columns:
        st.info("No data available for this view.")
        return

    df_liked = df_filtered.copy()
    df_liked = df_liked.sort_values("likescount", ascending=False)

    cols_to_show = []
    for col in ["likescount", "stars", "sentiment_from_stars", "review_text"]:
        if col in df_liked.columns:
            cols_to_show.append(col)

    st.dataframe(df_liked[cols_to_show].head(20), use_container_width=True)


# ---------------------------------------------------------
# 5. ENSEMBLE 3 MODEL LOADING & GENERIC PREDICTION
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_ensemble3_models(model_root: str = "models"):
    """
    Load the Stage 3 Ensemble 3 components:
    - DistilBERT baseline          (stage3_base_model)
    - TinyBERT SNSEM component     (stage3_snsem_tinybert)

    Returns:
        model_specs: list of (name, model, tokenizer)
        device: torch.device
        id2label: mapping {0: "negative", 1: "positive"} (fallback if missing)
    """
    base_path = os.path.join(model_root, "stage3_base_model")
    tinybert_path = os.path.join(model_root, "stage3_snsem_tinybert")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DistilBERT
    tok_distil = AutoTokenizer.from_pretrained(base_path)
    mdl_distil = AutoModelForSequenceClassification.from_pretrained(base_path)
    mdl_distil.to(device).eval()

    # TinyBERT
    tok_tiny = AutoTokenizer.from_pretrained(tinybert_path)
    mdl_tiny = AutoModelForSequenceClassification.from_pretrained(tinybert_path)
    mdl_tiny.to(device).eval()

    # Use DistilBERT's id2label if available; fallback to binary mapping
    id2label = getattr(
        mdl_distil.config,
        "id2label",
        {0: "negative", 1: "positive"},
    )

    model_specs = [
        ("distilbert_base", mdl_distil, tok_distil),
        ("tinybert_snsem", mdl_tiny, tok_tiny),
    ]

    return model_specs, device, id2label


def ensemble_predict_logits(
    texts: List[str],
    model_specs: List[Tuple[str, AutoModelForSequenceClassification, AutoTokenizer]],
    max_length: int = 256,
    batch_size: int = 16,
    device=None,
):
    """
    Run ensemble inference via logit averaging over multiple models.

    Args:
        texts: list of raw review texts
        model_specs: list of (name, model, tokenizer) tuples
        max_length: max sequence length for tokenization
        batch_size: batch size per forward pass
        device: torch.device or string ("cuda"/"cpu")

    Returns:
        np.ndarray of ensemble logits (num_samples, num_classes)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    all_logits = None

    for (name, model, tok) in model_specs:
        model.eval()

        model_logits = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            enc = tok(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc).logits.detach().cpu().numpy()
            model_logits.append(out)

        model_logits = np.vstack(model_logits)

        if all_logits is None:
            all_logits = model_logits
        else:
            all_logits += model_logits

    ensemble_logits = all_logits / len(model_specs)
    return ensemble_logits


def ensemble_predict_labels(
    texts: List[str],
    model_specs: List[Tuple[str, AutoModelForSequenceClassification, AutoTokenizer]],
    max_length: int = 256,
    batch_size: int = 16,
    device=None,
):
    """
    Convenience wrapper:
    - Computes ensemble logits
    - Applies softmax
    - Returns (predicted_labels, probabilities)
    """
    logits = ensemble_predict_logits(
        texts=texts,
        model_specs=model_specs,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    probs = softmax_np(logits, axis=1)   # shape: (n_samples, n_classes)
    labels = probs.argmax(axis=1)        # argmax over classes
    return labels, probs


def show_live_prediction_ensemble():
    """
    Streamlit UI wrapper for live sentiment prediction using Ensemble 3.

    Ensemble 3 = DistilBERT + TinyBERT, combined via logit averaging.
    """
    st.markdown("---")
    st.subheader("Demo: Analyze a New Review (Ensemble 3: DistilBERT + TinyBERT)")

    example_text = st.text_area(
        "Paste a guest review here (any language – the model expects English, but you can paste translations):",
        height=150,
        value="Amazing view and great service, but the food was a bit overpriced.",
    )

    if st.button("Analyze Review"):
        if not example_text.strip():
            st.warning("Please enter a review text before analyzing.")
            return

        with st.spinner("Running ensemble prediction..."):
            try:
                model_specs, device, id2label = load_ensemble3_models()
                # Note: texts is a list; we pass a single-element list here
                labels, probs = ensemble_predict_labels(
                    texts=[example_text],
                    model_specs=model_specs,
                    max_length=256,
                    batch_size=1,
                    device=device,
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return

        pred_id = int(labels[0])
        prob_vec = probs[0]   # shape: (num_classes,)

        label = id2label.get(pred_id, "unknown")

        # Assume binary: 0 = negative, 1 = positive
        if len(prob_vec) >= 2:
            prob_neg = float(prob_vec[0])
            prob_pos = float(prob_vec[1])
        else:
            # fallback if misconfigured
            prob_pos = float(prob_vec[pred_id])
            prob_neg = 1.0 - prob_pos

        st.markdown(f"**Predicted Sentiment (Ensemble 3): `{label}`**")

        # Progress bar for predicted class probability
        st.progress(float(prob_vec[pred_id]))

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.metric("P(Positive)", f"{prob_pos * 100:.2f}%")
        with col_neg:
            st.metric("P(Negative)", f"{prob_neg * 100:.2f}%")

        st.caption(
            "Prediction powered by the Stage 3 Transformer Ensemble 3 "
            "(DistilBERT + TinyBERT) using logit averaging, trained on "
            "balanced synthetic reviews and evaluated on the untouched "
            "Stage 1 real test set."
        )


# ---------------------------------------------------------
# 6. MAIN APP
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CN Tower 360 – Review Insights Dashboard",
        layout="wide",
    )

    st.title("CN Tower 360 – Review Insights Dashboard")
    st.write(
        "Business dashboard for **360 The Restaurant at the CN Tower** "
        "using real Google Reviews (Stage 1 dataset) and a Stage 3 Transformer "
        "Ensemble 3 (DistilBERT + TinyBERT) for live sentiment prediction."
    )

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.write(
        "- Data source: Google Reviews (real Stage 1 dataset, imbalanced)\n"
        "- Positive ≈ 4–5 stars, Negative ≈ 1–3 stars\n"
        "- Live predictor: Ensemble 3 (DistilBERT + TinyBERT, logit averaging)\n"
    )

    # Load data
    data_path = "data/stage1_business_full_clean.parquet"
    try:
        df = load_reviews(data_path)
    except Exception as e:
        st.error(f"Failed to load dataset from `{data_path}`: {e}")
        return

    # Apply filters via sidebar
    df_filtered = apply_filters(df)

    if df_filtered.empty:
        st.warning("No data matches the selected filters.")
    else:
        # KPI section
        show_kpis(df_filtered)

        # Charts layout: 2 x 2 grid (where bottom-right is recent negatives table)
        top_left, top_right = st.columns(2)
        with top_left:
            show_sentiment_over_time(df_filtered)
        with top_right:
            show_star_distribution(df_filtered)

        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            show_review_length_distribution(df_filtered)
        with bottom_right:
            show_recent_negative_reviews(df_filtered)

        # Full-width most liked table
        show_most_liked_reviews(df_filtered)

    # Live predictor at the bottom
    show_live_prediction_ensemble()


if __name__ == "__main__":
    main()
