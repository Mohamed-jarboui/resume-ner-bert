"""
Utilities for identifying and segmenting resume sections.
"""

import re
from typing import Dict, List, Optional

SECTION_MARKERS = {
    "EDUCATION": [
        r"education", r"formation", r"academic", r"cursus", r"étud", r"diplômes"
    ],
    "EXPERIENCE": [
        r"experience", r"expériences", r"profession", r"work", r"emploi", r"stage", r"parcours"
    ],
    "SKILLS": [
        r"skills", r"compétences", r"technique", r"expertise", r"stack", r"outils", r"tools"
    ],
    "PROJECTS": [
        r"projects", r"projets", r"accomplishments", r"réalisations"
    ],
    "LANGUAGES": [
        r"languages", r"langues", r"linguistique"
    ],
    "SUMMARY": [
        r"summary", r"profile", r"résumé", r"about", r"objectif", r"background"
    ]
}

def identify_section(line: str) -> Optional[str]:
    """Identify if a line is a section header."""
    line = line.strip().lower()
    if len(line) > 30: # Headers are usually short
        return None
        
    for section, markers in SECTION_MARKERS.items():
        for marker in markers:
            if re.search(r"^" + marker + r"[:\s\-]*$", line):
                return section
    return None

def segment_resume(text: str) -> Dict[str, str]:
    """Split resume text into sections."""
    lines = text.split('\n')
    sections = {"HEADER": ""}
    current_section = "HEADER"
    
    for line in lines:
        header = identify_section(line)
        if header:
            current_section = header
            sections[current_section] = ""
        else:
            sections[current_section] += line + "\n"
            
    return sections
