"""Custom Sphinx extension to handle mcreference tags."""

import re
from sphinx.application import Sphinx


def process_mcreference_tags(app, docname, source):
    """
    Process mcreference tags in the source and convert them to proper links.
    
    This function is called during the source-read event to transform
    mcreference tags into standard reStructuredText or Markdown links.
    """
    # Pattern to match mcreference tags
    pattern = r'<mcreference\s+link="([^"]+)"\s+index="([^"]+)">([^<]+)</mcreference>'
    
    def replace_mcreference(match):
        link = match.group(1)
        index = match.group(2)
        text = match.group(3)
        
        # Return as Markdown link format
        return f'[{text}]({link})'
    
    # Process each line in the source
    for i, line in enumerate(source):
        source[i] = re.sub(pattern, replace_mcreference, line)


def setup(app: Sphinx):
    """
    Setup function for the Sphinx extension.
    """
    # Connect the processing function to the source-read event
    app.connect('source-read', process_mcreference_tags)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }