#!/usr/bin/env python
"""
Final test to verify the scatter plot is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Testing Scatter Plot Fix ===")

# Test the import and basic functionality
try:
    from app.pipeline.glucose_charts import plot_scatter
    print("✓ Successfully imported plot_scatter")

    # Test basic functionality
    result = plot_scatter('2026-10-21')
    print(f"✓ Generated result of type: {type(result)}")

    # Check if it's a valid plotly figure
    import plotly.graph_objects as go
    if isinstance(result, go.Figure):
        print("✓ Result is a valid plotly Figure")
        print("✓ Scatter plot implementation is working")
    else:
        print("✗ Result is not a valid plotly Figure")
        print(f"✗ Result type: {type(result)}")

    # Try to generate HTML (this is what Gradio uses)
    if hasattr(result, 'to_html'):
        html = result.to_html()
        print(f"✓ HTML generation works, size: {len(html)} bytes")
    else:
        print("✗ No to_html method available")

except Exception as e:
    print(f"✗ Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Application Status ===")
try:
    from app.app import show_charts
    from app.storage import user_db
    import pandas as pd

    # Create a mock user
    mock_user = {
        'id': 1,
        'username': 'test_user',
        'diabetes_type': 'type2',
        'baseline_mgdl': 140.0,
        'created_at': '2026-10-15T08:00:00'
    }

    print("✓ Successfully imported show_charts function")

except Exception as e:
    print(f"✗ Error in app import: {e}")