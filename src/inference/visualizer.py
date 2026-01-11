"""
Enhanced Entity Visualizer for 7-entity schema.
"""

import streamlit as st
import re

class EntityVisualizer:
    """Helper class to visualize NER results with color-coded highlighting."""
    
    def __init__(self):
        # Define pretty colors for each entity type
        self.colors = {
            'NAME': '#e67e22',        # Orange
            'EMAIL': '#9b59b6',       # Purple
            'LOCATION': '#f1c40f',    # Yellow
            'DATE': '#1abc9c',        # Turquoise
            'DEGREE': '#2ecc71',      # Emerald Green
            'SKILL': '#3498db',       # Peter River Blue
            'EXPERIENCE': '#e74c3c',  # Alizarin Red
            'DEFAULT': '#95a5a6'      # Concrete Grey
        }

    def get_html(self, text: str, entities: list) -> str:
        """Generate HTML string with highlighted entities."""
        # Sort entities by start position in reverse to avoid index shifting
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
        
        html = text
        for ent in sorted_entities:
            # Skip entities without position info
            if 'start' not in ent or 'end' not in ent:
                continue
                
            start, end = ent['start'], ent['end']
            etype = ent['type']
            color = self.colors.get(etype, self.colors['DEFAULT'])
            
            # Build highlighted span
            highlight = (
                f'<span style="background-color: {color}; color: white; '
                f'padding: 2px 6px; border-radius: 4px; font-weight: bold; '
                f'margin: 0 2px; cursor: help;" title="{etype}">'
                f'{html[start:end]}'
                f'<small style="font-size: 0.6em; margin-left: 4px; opacity: 0.8;">{etype}</small>'
                f'</span>'
            )
            
            # Replace in text
            html = html[:start] + highlight + html[end:]
            
        # Wrap in a div with proper line height
        return f'<div style="line-height: 2.5; font-family: sans-serif; padding: 20px; background: #fdfdfd; border-radius: 8px; border: 1px solid #eee;">{html}</div>'

    def display_legend(self):
        """Display a legend for entity colors in Streamlit."""
        st.write("### 🏷️ Entity Legend")
        cols = st.columns(len(self.colors) - 1)
        for i, (etype, color) in enumerate(self.colors.items()):
            if etype == 'DEFAULT': continue
            cols[i % len(cols)].markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 4px; margin-right: 8px;"></div>'
                f'<span>{etype}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
