"""
fix_pages.py
Removes st.set_page_config(...) calls from all page files
since it is already defined once in app.py
"""

import os
import re

pages_dir = os.path.join(os.path.dirname(__file__), "pages")

# Pattern to match st.set_page_config(...) — possibly multiline
pattern = re.compile(
    r'st\.set_page_config\s*\([^)]*\)\s*\n?',
    re.DOTALL
)

for filename in os.listdir(pages_dir):
    if filename.endswith(".py"):
        filepath = os.path.join(pages_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = pattern.sub("", content)

        if new_content != content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"✅ Fixed: {filename}")
        else:
            print(f"⏭️  No change needed: {filename}")

print("\nDone! All page files have been processed.")