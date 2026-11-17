from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parent.parent
DASHBOARD = ROOT / "dashboard" / "app.py"
PAGES = [
    "📊 Overview",
    "🗂️ Data",
    "📏 Traditional Metrics",
    "🧠 G-Eval",
    "🏆 Model Rankings",
    "⚖️ Bias Analysis",
    "🎛️ Unified Score / Human Alignment",
    "🚀 Live G-Eval",
]


def test_all_dashboard_pages_render():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=30).run()

    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page} raised an exception"
        assert not app.error, f"{page} rendered a red error"
        assert not any("MockLLMClient" in element.value for element in app.markdown)
        assert any("Results dir:" in caption.value for caption in app.sidebar.caption)

        if page == "🎛️ Unified Score / Human Alignment":
            weights = [element.value for element in app.slider]
            assert weights == [0.4, 0.25, 0.15, 0.2]
            assert not any("Weights sum to" in warning.value for warning in app.warning)
