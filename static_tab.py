import streamlit as st
import pandas as pd
import base64

def show_static_results():
    # --- PNGs ---
    st.markdown("### Key Drivers of Satisfaction and Complaints as of December 2025")

    plot_files = [
        ("Sentiment by Stars", "static_assets/stage5_sentiment_by_stars_explained.png"),
        ("Negative Topics", "static_assets/stage5_negative_topics.png"),
        ("Negative Keywords", "static_assets/stage5_negative_keywords.png"),
        ("Topics by Stars", "static_assets/stage5_topics_by_stars.png"),
    ]

    cols = st.columns(2)
    for i, (caption, path) in enumerate(plot_files):
        with cols[i % 2]:
            try:
                with open(path, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode()
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f9f9f9;
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        border: 1px solid #ddd;
                    ">
                        <h4 style="margin-top:0;">{caption}</h4>
                        <img src="data:image/png;base64,{encoded}" style="width:100%; border-radius:6px;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Could not load {caption}: {e}")

    st.caption(
        "These visual insights are based on sentiment analysis and topic modeling from customer reviews. "
        "To update this section, replace the image files in the `static_assets/` folder."
    )

    # --- CSVs ---
    st.markdown("### Customer Review Topics (Detailed View)")

    csv_files = {
        "All Topics Across Reviews": "static_assets/stage5_topics.csv",
        "Topics Split by Star Rating": "static_assets/stage5_topics_by_stars.csv",
    }

    for label, path in csv_files.items():
        try:
            st.markdown(f"**{label}**")
            df = pd.read_csv(path)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load {label}: {e}")
