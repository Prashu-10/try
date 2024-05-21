# try

import re

def split_markdown(markdown):
    # Regular expression to match headers (e.g., # Header, ## Header, ### Header, etc.)
    pattern = re.compile(r'(?m)^#{1,6} .*$')
    sections = pattern.split(markdown)
    headers = pattern.findall(markdown)
    
    # Combine headers and sections
    result = []
    for i, section in enumerate(sections):
        if i < len(headers):
            result.append(headers[i] + section)
        else:
            result.append(section)
    
    return result

# Example usage
markdown_text = """
# Section 1
Content of section 1.

## Subsection 1.1
Content of subsection 1.1.

# Section 2
Content of section 2.
"""

sections = split_markdown(markdown_text)
for i, section in enumerate(sections):
    print(f"Section {i+1}:\n{section}\n")