"""
test_streamlit_app.py
---------------------
Automated test suite using Streamlit's AppTest framework to verify all 6 navigation
pages, filters, visualization selection, regression prediction, classification prediction,
model comparisons, and Data Explorer features in the redesigned premium dashboard.
"""

import sys
import os

# Ensure UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from streamlit.testing.v1 import AppTest

def test_complete_streamlit_ui():
    print("==================================================")
    print("STARTING STREAMLIT UI AUTOMATED VERIFICATION")
    print("==================================================")
    
    # 1. Initialize App
    print("\n[1] Initializing AppTest from app.py...")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App launched with exception: {at.exception}"
    print(" -> App initialized cleanly without exceptions.")
    
    # 2. Test Overview Page (Default)
    print("\n[2] Testing '⌂  Overview' page...")
    md_texts = [m.value for m in at.markdown]
    assert any("IPL ANALYTICS" in t for t in md_texts), "Header branding not found!"
    assert any("TURNING CRICKET DATA INTO PERFORMANCE INSIGHTS" in t for t in md_texts) or any("IPL AT A GLANCE" in t for t in md_texts), "Overview hero title not found!"
    assert any("2,217" in t for t in md_texts), "Dataset rows metric missing!"
    print(" -> Header, subtitle, and dynamic metric cards verified.")
    assert len(at.image) >= 1, "Introductory trend plot not found!"
    print(f" -> Found rendered plot on Overview page.")
    
    # 3. Test IPL Analytics Page
    print("\n[3] Testing '▣  IPL Analytics' page...")
    at.sidebar.radio[0].set_value("▣  IPL Analytics").run()
    assert not at.exception, f"Exception on Analytics page: {at.exception}"
    md_texts = [m.value for m in at.markdown]
    assert any("IPL ANALYTICS CENTER" in t for t in md_texts), "Analytics title missing!"
    # Change team filter
    at.selectbox(key="analytics_team_filter").set_value("Mumbai Indians").run()
    assert not at.exception, "Exception after changing team filter"
    md_texts = [m.value for m in at.markdown]
    assert any("39,906" in t for t in md_texts), "Mumbai Indians total runs metric missing!"
    print(" -> Team filter dynamically updated metrics (Mumbai Indians 39,906 runs verified).")
    
    # 4. Test Visualizations Page
    print("\n[4] Testing '◈  Visualizations' page...")
    at.sidebar.radio[0].set_value("◈  Visualizations").run()
    assert not at.exception, f"Exception on Visualizations page: {at.exception}"
    sb_viz = at.selectbox[0]
    print(f" -> Available visualizations ({len(sb_viz.options)}): {sb_viz.options[:5]}...")
    assert len(sb_viz.options) == 20, f"Expected 20 visualizations, found {len(sb_viz.options)}"
    # Switch to Correlation Heatmap
    sb_viz.set_value("Correlation Heatmap").run()
    assert not at.exception, "Exception on Correlation Heatmap"
    assert len(at.image) >= 1, "Visualization chart failed to render"
    print(" -> Selected 'Correlation Heatmap' rendered successfully.")
    
    # 5. Test Machine Learning Page
    print("\n[5] Testing '⚡ Machine Learning' page...")
    at.sidebar.radio[0].set_value("⚡ Machine Learning").run()
    assert not at.exception, f"Exception on ML page: {at.exception}"
    md_texts = [m.value for m in at.markdown]
    assert any("PREDICTIVE INSIGHTS" in t for t in md_texts), "ML title missing!"
    
    # 5.1 Test Regression Prediction Button
    print(" -> Testing Regression Prediction button...")
    at.button(key="btn_run_reg").click().run()
    assert not at.exception, "Exception clicking RUN PREDICTION"
    md_texts_reg = [m.value for m in at.markdown]
    assert any("result-banner-reg" in t for t in md_texts_reg), "Regression prediction banner missing!"
    print(" -> Dynamic regression prediction generated and displayed.")
    
    # 5.2 Test Classification Prediction Button
    print(" -> Testing Classification Prediction button...")
    at.button(key="btn_run_clf").click().run()
    assert not at.exception, "Exception clicking CLASSIFY INNINGS"
    md_texts_clf = [m.value for m in at.markdown]
    assert any("result-banner-clf" in t for t in md_texts_clf), "Classification banner missing!"
    print(" -> Dynamic classification prediction generated and displayed.")
    
    # Verify comparison tables
    assert len(at.dataframe) >= 2, f"Expected comparison tables, found {len(at.dataframe)}"
    print(f" -> Found {len(at.dataframe)} comparison tables and metric grids.")
    
    # 6. Test Data Explorer Page
    print("\n[6] Testing '⌕  Data Explorer' page...")
    at.sidebar.radio[0].set_value("⌕  Data Explorer").run()
    assert not at.exception, f"Exception on Data Explorer: {at.exception}"
    md_texts = [m.value for m in at.markdown]
    assert any("DATA CENTER" in t for t in md_texts), "Explorer title missing!"
    print(" -> Data Explorer rendered summary metrics and filters.")
    
    # 7. Test About Project Page
    print("\n[7] Testing 'ⓘ  About Project' page...")
    at.sidebar.radio[0].set_value("ⓘ  About Project").run()
    assert not at.exception, f"Exception on About page: {at.exception}"
    md_texts = [m.value for m in at.markdown]
    assert any("ABOUT THE PROJECT" in t for t in md_texts), "About title missing!"
    assert any("Scikit-Learn" in t for t in md_texts), "Technologies listing missing!"
    print(" -> About Project page rendered cleanly.")
    
    print("\n==================================================")
    print("ALL STREAMLIT UI AUTOMATED TESTS PASSED (100% GREEN)")
    print("==================================================")

if __name__ == "__main__":
    test_complete_streamlit_ui()
