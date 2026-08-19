import streamlit as st

def inject_custom_css() -> None:
    """Inject custom premium CSS into the Streamlit page."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, p, span, li, label {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Backgrounds */
    .stApp {
        background-color: #0b0f17;
        color: #e6edf3;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e131f 0%, #080b12 100%);
        border-right: 1px solid #1f293d;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e !important;
        font-size: 0.95rem;
    }

    /* Headers with gradient background */
    h1 {
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2, h3 {
        background: linear-gradient(135deg, #79c0ff 0%, #d8b4fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.0rem !important;
    }

    /* Custom Cards styling for Expanders */
    div[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1f293d !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.2) !important;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(137, 87, 229, 0.4) !important;
        background: linear-gradient(135deg, #388bfd 0%, #ab7df8 100%) !important;
        color: #ffffff !important;
    }

    div.stButton > button:first-child:active {
        transform: translateY(1px) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button:first-child {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.2) !important;
    }

    div.stDownloadButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(46, 160, 67, 0.4) !important;
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
        color: #ffffff !important;
    }

    /* Selectboxes and Inputs */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }

    /* Metric elements styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161f30 0%, #0d131f 100%);
        border: 1px solid #1f2f4d;
        border-radius: 12px;
        padding: 1.2rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 2.0rem !important;
        font-weight: 700 !important;
    }

    /* Dataframe borders and shadows */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Divider color */
    hr {
        border-color: #21262d !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "") -> None:
    """Render a premium glassmorphic metric card using HTML/CSS.
    
    Use this when st.metric is not sufficient or when we want extra custom details.
    """
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(22, 31, 48, 0.7) 0%, rgba(13, 19, 31, 0.7) 100%);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        margin-bottom: 1rem;
    ">
        <div style="color: #8b949e; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.5rem;">
            {title}
        </div>
        <div style="color: #58a6ff; font-size: 2.2rem; font-weight: 700; line-height: 1.1;">
            {value}
        </div>
        {f'<div style="color: #8b949e; font-size: 0.8rem; margin-top: 0.5rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
