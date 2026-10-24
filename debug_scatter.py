#!/usr/bin/env python
"""Debug script for scatter plot issue"""

import sys
sys.path.append('.')

from app.pipeline.glucose_charts import plot_scatter, _empty
import pandas as pd
import traceback

print("Testing scatter plot generation...")

try:
    # Try to generate a scatter plot
    result = plot_scatter('2026-10-21')

    print(f"Result type: {type(result)}")

    if hasattr(result, 'to_html'):
        print("Successfully generated HTML")
        html = result.to_html()
        print(f"HTML length: {len(html)}")
    else:
        print("Result doesn't have to_html method")
        print(f"Result: {result}")

    # Test with empty result
    empty_result = _empty("Test", "Test message")
    print(f"Empty result type: {type(empty_result)}")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()