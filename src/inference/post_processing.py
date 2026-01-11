import re
from typing import List, Dict

class SkillFilter:
    """Strict post-processing filters for Resume NER to eliminate false positives in SKILL tags."""
    
    def __init__(self):
        # Regex patterns for common misclassifications
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.date_pattern = re.compile(r'\b(19|20)\d{2}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', re.I)
        
        # Blacklist of words that are frequently mislabeled as skills but are definitely not
        self.o_blacklist = {
            "passionné", "reconnu", "capacité", "solutions", "performantes",
            "motivated", "excellent", "proven", "track", "record", "highly",
            "experience", "years", "months", "responsible", "working", "team"
        }

    def filter_entities(self, entities: List[Dict]) -> List[Dict]:
        """Apply filters to a list of extracted entities."""
        filtered = []
        
        for ent in entities:
            etype = ent['type']
            text = ent['entity']
            text_lower = text.lower().strip()
            
            # 1. Rule: Emails are NEVER skills
            if etype == "SKILL" and self.email_pattern.search(text):
                continue
                
            # 2. Rule: Numbers alone or dates are NEVER skills
            if etype == "SKILL" and (text.isdigit() or self.date_pattern.search(text)):
                continue
                
            # 3. Rule: Blacklisted generic words are NEVER skills
            if etype == "SKILL" and text_lower in self.o_blacklist:
                continue
                
            # 4. Rule: Very short "skills" are often noise (unless they are known abbreviations like 'C', 'R', 'Go')
            if etype == "SKILL" and len(text_lower) < 2 and text_lower not in ['c', 'r', 'go']:
                continue
            
            # 5. Rule: Locations appearing in SKILL (heuristic check)
            # If the word consists of common address/city patterns, we might drop it if labeled SKILL
            
            filtered.append(ent)
            
        return filtered

    def normalize_results(self, entities: List[Dict]) -> Dict:
        """Group and deduplicate into the final expected JSON structure."""
        output = {
            "name": "",
            "email": "",
            "location": "",
            "degrees": [],
            "skills": [],
            "experience": []
        }
        
        seen_skills = set()
        seen_degrees = set()
        seen_exp = set()
        
        for ent in entities:
            etype = ent['type']
            text = ent['entity'].strip()
            
            if etype == "NAME" and not output["name"]:
                output["name"] = text
            elif etype == "EMAIL" and not output["email"]:
                output["email"] = text
            elif etype == "LOCATION" and not output["location"]:
                output["location"] = text
            elif etype == "DEGREE":
                if text.lower() not in seen_degrees:
                    output["degrees"].append(text)
                    seen_degrees.add(text.lower())
            elif etype == "SKILL":
                if text.lower() not in seen_skills:
                    output["skills"].append(text)
                    seen_skills.add(text.lower())
            elif etype == "EXPERIENCE" or etype == "DATE":
                # Often dates and roles are part of experience
                if text.lower() not in seen_exp:
                    output["experience"].append(text)
                    seen_exp.add(text.lower())
                    
        return output
