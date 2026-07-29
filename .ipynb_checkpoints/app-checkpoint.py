import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.metrics.pairwise import linear_kernel

# Page Configuration
st.set_page_config(
    page_title="Amazon Product Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Session State Initialization (Wishlist)
# ------------------------------------------------------------------------------
if "wishlist" not in st.session_state:
    st.session_state.wishlist = []

# ------------------------------------------------------------------------------
# Targeted CSS Rules (UI Consistency Locks)
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Open Sans', Arial, sans-serif !important;
    }

    /* Force Light Background on Main App View */
    section[data-testid="stMain"] {
        background-color: #eaeded !important;
    }

    /* Force Light Theme Text Default across Main Body */
    section[data-testid="stMain"] p, 
    section[data-testid="stMain"] label, 
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] div {
        color: #0f1111;
    }

    /* --- 1. Dark Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: #232f3e !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* --- 2. White Card Containers --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        padding: 16px !important;
        border: 1px solid #d5d9d9 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
    }

    /* Strip background styling from block wrappers that contain the Header Banner */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.amazon-header-banner),
    div[data-testid="stElementContainer"]:has(.amazon-header-banner) {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* --- 3. Amazon Header Banner Styling --- */
    .amazon-header-banner {
        background-color: #131921 !important;
        padding: 18px 25px !important;
        border-radius: 8px !important;
        margin-bottom: 20px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2) !important;
    }
    .amazon-header-banner * {
        color: #ffffff !important;
        background: transparent !important;
    }

    /* --- 4. Metric KPI Cards --- */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 1px solid #d5d9d9 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #565959 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] * {
        color: #0f1111 !important;
        font-weight: 700 !important;
    }

    /* --- 5. Action Buttons --- */
    div.stButton > button {
        background-color: #ffd814 !important;
        color: #0f1111 !important;
        border: 1px solid #fcd200 !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 5px rgba(213,217,217,0.5) !important;
        width: 100% !important;
    }
    div.stButton > button * {
        color: #0f1111 !important;
    }
    div.stButton > button:hover {
        background-color: #f7ca00 !important;
        border-color: #f2c200 !important;
    }

    /* --- 6. Link Buttons (Single Outer Pill) --- */
    [data-testid="stLinkButton"] {
        width: 100% !important;
    }
    [data-testid="stLinkButton"] a {
        background-color: #ffa41c !important;
        border: 1px solid #ff8f00 !important;
        border-radius: 20px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        text-decoration: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 8px 16px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    [data-testid="stLinkButton"] a:hover {
        background-color: #e8920d !important;
    }
    [data-testid="stLinkButton"] a * {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: #0f1111 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* --- 7. Inputs & Selectors --- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border-color: #888c8c !important;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] * {
        color: #0f1111 !important;
    }

    /* Badges */
    .price-text {
        color: #b12704 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    .rating-badge {
        color: #de7921 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Data Loading & Cleaning
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    products = pickle.load(open("products.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    indices = pickle.load(open("indices.pkl", "rb"))

    # Cleaning Numerical Columns
    products["ratings"] = pd.to_numeric(products["ratings"], errors="coerce")
    
    products["discount_price"] = (
        products["discount_price"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    products["discount_price"] = pd.to_numeric(products["discount_price"], errors="coerce")

    products["no_of_ratings"] = (
        products["no_of_ratings"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    products["no_of_ratings"] = pd.to_numeric(products["no_of_ratings"], errors="coerce")

    return products, tfidf, tfidf_matrix, indices

products, tfidf, tfidf_matrix, indices = load_and_clean_data()

# ------------------------------------------------------------------------------
# Hybrid Recommendation Engine Logic
# ------------------------------------------------------------------------------
def hybrid_recommend_products(product_name, top_n=6, min_rating=0.0, max_price=None):
    idx = indices[product_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
        
    cosine_scores = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    
    df = products.copy()
    df["similarity_score"] = cosine_scores
    
    # Exclude selected item
    df = df[df.index != idx]
    
    # Apply user side filters
    if min_rating > 0:
        df = df[df["ratings"] >= min_rating]
    if max_price is not None:
        df = df[df["discount_price"] <= max_price]
        
    # Hybrid calculation: 70% content similarity + 30% rating weight
    norm_ratings = df["ratings"].fillna(df["ratings"].mean()) / 5.0
    df["hybrid_score"] = (df["similarity_score"] * 0.7) + (norm_ratings * 0.3)
    
    top_recs = df.sort_values(by="hybrid_score", ascending=False).head(top_n)
    return top_recs

# ------------------------------------------------------------------------------
# App Header Banner
# ------------------------------------------------------------------------------
st.markdown("""
    <div class="amazon-header-banner">
        <div>
            <h1 style="color: #ffffff !important; font-size: 24px; font-weight: 700; margin: 0; padding: 0;">
                amazon<span style="color: #ff9900 !important;">.in</span> Recommendation Engine
            </h1>
            <div style="color: #cccccc !important; font-size: 13px; margin-top: 4px;">
                Personalized Recommendations & User Behavior Analytics
            </div>
        </div>
        <div style="font-size: 28px; color: #ffffff !important;">🛒</div>
    </div>
""", unsafe_allow_html=True)

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Products", f"{len(products):,}")
c2.metric("Categories", products["main_category"].nunique())
c3.metric("Avg Rating", f"{round(products['ratings'].mean(), 2)} ★")
c4.metric("Avg Price", f"₹{round(products['discount_price'].mean(), 2):,}")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar Navigation & Filters
# ------------------------------------------------------------------------------
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to", 
    [
        "Recommendation Engine", 
        f"❤️ My Wishlist ({len(st.session_state.wishlist)})", 
        "Analytics Dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Recommendation Filters")

# Filter Controls
max_price_limit = float(products["discount_price"].max()) if pd.notna(products["discount_price"].max()) else 50000.0
min_rating_filter = st.sidebar.slider("Min Product Rating (★)", 0.0, 5.0, 3.5, step=0.5)
max_price_filter = st.sidebar.slider("Max Budget (₹)", 100.0, max_price_limit, max_price_limit, step=500.0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Developed by BBD Students:**
* Shikha Yadav
* Shivam Shah
* Shubham Chauhan
""")

# ------------------------------------------------------------------------------
# Page 1: Recommendation Engine
# ------------------------------------------------------------------------------
if page == "Recommendation Engine":
    
    # Keyword Search & Dropdown Combo
    search_term = st.text_input("🔍 Search product name:", placeholder="Type to filter list (e.g. pouch, cable, bag)...")
    
    if search_term:
        filtered_names = [name for name in products["name"].unique() if search_term.lower() in str(name).lower()]
        if not filtered_names:
            st.warning("No products matched your search keyword. Showing all products instead.")
            filtered_names = products["name"].unique()
    else:
        filtered_names = products["name"].unique()
        
    selected_product = st.selectbox("Select a Product:", filtered_names)
    product = products[products["name"] == selected_product].iloc[0]
    
    # Main Product Card
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])

        with col1:
            try:
                st.image(product["image"], width=220)
            except Exception:
                st.warning("No Image Available")

        with col2:
            st.markdown(
                f"<div style='color: #0f1111 !important; font-size: 20px !important; font-weight: 700 !important; line-height: 1.3 !important; margin-bottom: 12px !important;'>{product['name']}</div>",
                unsafe_allow_html=True
            )
            
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"<p style='color: #0f1111 !important;'><b>Category:</b> {product['main_category']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #0f1111 !important;'><b>Sub-Category:</b> {product['sub_category']}</p>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<p style='color: #0f1111 !important;'><b>Rating:</b> <span class='rating-badge'>{product['ratings']} ★</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #0f1111 !important;'><b>Price:</b> <span class='price-text'>₹{product['discount_price']:,}</span></p>", unsafe_allow_html=True)

            st.write("")
            
            b1, b2 = st.columns(2)
            with b1:
                if "link" in products.columns and pd.notna(product["link"]):
                    st.link_button("🛒 View on Amazon", product["link"], use_container_width=True)
            with b2:
                if st.button("❤️ Save to Wishlist", key="main_wishlist"):
                    if product["name"] not in [item["name"] for item in st.session_state.wishlist]:
                        st.session_state.wishlist.append(product.to_dict())
                        st.success("Saved to Wishlist!")
                    else:
                        st.info("Item already in Wishlist.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate Recommendations", type="primary"):
        recommendations = hybrid_recommend_products(
            selected_product, 
            top_n=6, 
            min_rating=min_rating_filter, 
            max_price=max_price_filter
        )
        
        st.subheader("Recommended Products For You (Hybrid Ranked)")
        
        if len(recommendations) == 0:
            st.warning("No recommendations met your rating and price filter criteria. Try adjusting the sidebar filters.")
        else:
            cols = st.columns(2)
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with cols[i % 2]:
                    with st.container(border=True):
                        c_img, c_desc = st.columns([1, 2])
                        with c_img:
                            try:
                                st.image(row["image"], width=130)
                            except Exception:
                                st.write("No Image Available")
                        
                        with c_desc:
                            st.markdown(
                                f"<div style='color: #0f1111 !important; font-weight: 700 !important; font-size: 14px !important; line-height: 1.3 !important; margin-bottom: 8px !important;'>{row['name'][:55]}...</div>",
                                unsafe_allow_html=True
                            )
                            st.markdown(f"<div style='margin-bottom: 4px;'><span class='rating-badge'>{row['ratings']} ★</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='margin-bottom: 6px;'><span class='price-text'>₹{row['discount_price']:,}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='color: #565959 !important; font-size: 12px;'>Category: {row['main_category']}</div>", unsafe_allow_html=True)

                        rb1, rb2 = st.columns(2)
                        with rb1:
                            if "link" in products.columns and pd.notna(row["link"]):
                                st.link_button("View Product", row["link"], use_container_width=True)
                        with rb2:
                            if st.button("❤️ Save", key=f"rec_wishlist_{i}"):
                                if row["name"] not in [item["name"] for item in st.session_state.wishlist]:
                                    st.session_state.wishlist.append(row.to_dict())
                                    st.success("Added!")
                                else:
                                    st.info("Already saved!")

# ------------------------------------------------------------------------------
# Page 2: My Wishlist Page
# ------------------------------------------------------------------------------
elif "Wishlist" in page:
    st.header("❤️ Saved Wishlist")
    
    if len(st.session_state.wishlist) == 0:
        st.info("Your wishlist is currently empty. Click '❤️ Save to Wishlist' on any product to save items here!")
    else:
        if st.button("🗑️ Clear Entire Wishlist"):
            st.session_state.wishlist = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, item in enumerate(st.session_state.wishlist):
            with st.container(border=True):
                col_w1, col_w2, col_w3 = st.columns([1, 3, 1])
                with col_w1:
                    try:
                        st.image(item["image"], width=100)
                    except Exception:
                        st.write("No Image")
                with col_w2:
                    st.markdown(f"**{item['name']}**")
                    st.markdown(f"<span class='rating-badge'>{item['ratings']} ★</span> | <span class='price-text'>₹{item['discount_price']:,}</span>", unsafe_allow_html=True)
                    st.caption(f"Category: {item['main_category']}")
                with col_w3:
                    if "link" in item and pd.notna(item["link"]):
                        st.link_button("Buy on Amazon", item["link"], key=f"wish_buy_{idx}")
                    if st.button("Remove", key=f"wish_rem_{idx}"):
                        st.session_state.wishlist.pop(idx)
                        st.rerun()

# ------------------------------------------------------------------------------
# Page 3: Interactive Analytics Dashboard (Expanded Plotly Suite)
# ------------------------------------------------------------------------------
else:
    st.header("📊 Interactive Product & Market Analytics")

    # Row 1: Category Count & Rating Distribution
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        with st.container(border=True):
            st.subheader("Top Main Categories")
            top_cats = products["main_category"].value_counts().head(10).reset_index()
            top_cats.columns = ["Category", "Count"]
            
            fig_cat = px.bar(
                top_cats, 
                x="Category", 
                y="Count", 
                color_discrete_sequence=["#232f3e"]
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_chart2:
        with st.container(border=True):
            st.subheader("Ratings Distribution")
            fig_ratings = px.histogram(
                products.dropna(subset=["ratings"]), 
                x="ratings", 
                nbins=20, 
                color_discrete_sequence=["#ffa41c"]
            )
            fig_ratings.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

    # Row 2: Price vs. Rating Scatter & Average Price per Category
    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        with st.container(border=True):
            st.subheader("Price vs. Rating Correlation")
            clean_scatter = products.dropna(subset=["discount_price", "ratings"]).head(1000)
            fig_scatter = px.scatter(
                clean_scatter,
                x="discount_price",
                y="ratings",
                color="main_category",
                hover_data=["name"],
                labels={"discount_price": "Price (₹)", "ratings": "Rating ★", "main_category": "Category"},
                opacity=0.75
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col_chart4:
        with st.container(border=True):
            st.subheader("Average Price by Main Category")
            avg_price = products.groupby("main_category")["discount_price"].mean().reset_index()
            avg_price = avg_price.sort_values(by="discount_price", ascending=False).head(10)
            fig_avg_price = px.bar(
                avg_price,
                x="discount_price",
                y="main_category",
                orientation="h",
                color_discrete_sequence=["#008296"],
                labels={"discount_price": "Avg Price (₹)", "main_category": "Category"}
            )
            fig_avg_price.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_avg_price, use_container_width=True)

    # Row 3: Top Sub-Categories Breakdown
    with st.container(border=True):
        st.subheader("Top 10 Sub-Categories Breakdown")
        top_sub = products["sub_category"].value_counts().head(10).reset_index()
        top_sub.columns = ["Sub-Category", "Count"]
        fig_sub = px.bar(
            top_sub,
            x="Sub-Category",
            y="Count",
            color="Sub-Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_sub.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    # Row 4: Price Distribution Box Plot
    with st.container(border=True):
        st.subheader("Category Price Distribution Spread")
        fig_price = px.box(
            products.dropna(subset=["discount_price"]), 
            x="main_category", 
            y="discount_price",
            color_discrete_sequence=["#232f3e"]
        )
        fig_price.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Category",
            yaxis_title="Price (₹)"
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # Row 5: Data Tables
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.subheader("Top Rated Products")
            top_rated = products.sort_values(by="ratings", ascending=False).head(10)
            st.dataframe(top_rated[["name", "ratings", "discount_price"]], use_container_width=True)

    with col_b:
        with st.container(border=True):
            st.subheader("Most Reviewed Products")
            most_reviewed = products.sort_values(by="no_of_ratings", ascending=False).head(10)
            st.dataframe(most_reviewed[["name", "no_of_ratings", "ratings"]], use_container_width=True)

    with st.container(border=True):
        st.subheader("Dataset Summary Statistics")
        st.write(products.describe())

# Footer
st.markdown("---")
st.caption("Amazon Recommendation Engine System | Built with Python, NLP, and Streamlit")