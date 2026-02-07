"""Content quality analyzer for resume content assessment."""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ContentQualityScore:
    """Content quality scoring results."""
    overall_score: int = 0
    action_verb_score: int = 0
    quantification_score: int = 0
    bullet_structure_score: int = 0
    conciseness_score: int = 0
    
    action_verbs_found: List[str] = field(default_factory=list)
    weak_verbs_found: List[str] = field(default_factory=list)
    quantified_achievements: List[str] = field(default_factory=list)
    bullet_points: List[Dict[str, Any]] = field(default_factory=list)
    
    recommendations: List[str] = field(default_factory=list)


class ContentAnalyzer:
    """Analyze resume content quality."""
    
    # Strong action verbs organized by category
    STRONG_ACTION_VERBS = {
        'leadership': [
            'led', 'managed', 'directed', 'supervised', 'coordinated', 'orchestrated',
            'spearheaded', 'headed', 'chaired', 'governed', 'guided', 'piloted',
            'commanded', 'administered', 'controlled', 'overseen'
        ],
        'achievement': [
            'achieved', 'accomplished', 'delivered', 'attained', 'obtained', 'secured',
            'earned', 'fulfilled', 'realized', 'exceeded', 'surpassed', 'outperformed'
        ],
        'development': [
            'developed', 'created', 'built', 'designed', 'engineered', 'architected',
            'constructed', 'established', 'formed', 'generated', 'produced', 'launched',
            'initiated', 'founded', 'implemented', 'deployed', 'introduced'
        ],
        'improvement': [
            'improved', 'enhanced', 'optimized', 'upgraded', 'refined', 'streamlined',
            'transformed', 'revolutionized', 'modernized', 'strengthened', 'boosted',
            'elevated', 'maximized', 'increased', 'accelerated', 'expanded'
        ],
        'problem_solving': [
            'solved', 'resolved', 'fixed', 'addressed', 'handled', 'corrected',
            'remediated', 'troubleshot', 'debugged', 'investigated', 'analyzed',
            'diagnosed', 'researched', 'identified', 'discovered', 'uncovered'
        ],
        'communication': [
            'communicated', 'presented', 'conveyed', 'articulated', 'negotiated',
            'persuaded', 'influenced', 'collaborated', 'liaised', 'mediated',
            'facilitated', 'moderated', 'translated', 'documented', 'authored'
        ],
        'technical': [
            'programmed', 'coded', 'developed', 'engineered', 'architected', 'configured',
            'integrated', 'automated', 'processed', 'analyzed', 'tested', 'debugged',
            'maintained', 'monitored', 'optimized', 'secured', 'migrated'
        ],
        'analysis': [
            'analyzed', 'evaluated', 'assessed', 'audited', 'examined', 'inspected',
            'reviewed', 'measured', 'quantified', 'calculated', 'compared', 'benchmarked',
            'researched', 'studied', 'surveyed', 'tested'
        ],
        'operations': [
            'operated', 'executed', 'performed', 'conducted', 'carried out', 'delivered',
            'processed', 'handled', 'managed', 'administered', 'maintained', 'supported',
            'serviced', 'facilitated', 'enabled'
        ],
        'financial': [
            'budgeted', 'forecasted', 'projected', 'allocated', 'reduced', 'saved',
            'cut', 'decreased', 'invested', 'funded', 'financed', 'audited',
            'monetized', 'profited', 'grew'
        ]
    }
    
    # Flatten all strong verbs for detection
    ALL_STRONG_VERBS = [verb for verbs in STRONG_ACTION_VERBS.values() for verb in verbs]
    
    # Weak verbs to avoid
    WEAK_VERBS = [
        'helped', 'assisted', 'worked on', 'participated', 'involved in',
        'responsible for', 'handled', 'did', 'made', 'got', 'took care of',
        'was', 'were', 'had', 'have', 'being', 'been',
        'supported', 'aided', 'contributed to', 'collaborated on'
    ]
    
    # Quantification patterns
    QUANTIFICATION_PATTERNS = {
        'percentage': [
            r'\b\d{1,3}%',
            r'\b\d{1,3}\s*percent',
            r'\d{1,3}\s*pct',
            r'increased by \d+',
            r'decreased by \d+',
            r'reduced by \d+',
            r'improved by \d+',
            r'grew by \d+',
        ],
        'monetary': [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\$\d+\s*(?:k|K|thousand|million|M|billion|B)',
            r'\d+\s*(?:k|K|thousand)\s*(?:dollars|USD)?',
            r'\d+\s*(?:million|M)\s*(?:dollars|USD)?',
        ],
        'count': [
            r'\b\d+\s*(?:users?|customers?|clients?|people|employees?|team members?)',
            r'\b\d+\s*(?:projects?|products?|features?|systems?|applications?)',
            r'\b\d+\s*(?:reports?|presentations?|documents?|meetings?)',
        ],
        'time': [
            r'\b\d+\s*(?:months?|years?|weeks?|days?|hours?)',
            r'\d+\+?\s*years?\s*(?:of)?\s*experience',
            r'over \d+\s*(?:months?|years?)',
        ],
        'scale': [
            r'\b\d+\s*(?:x|times?)',
            r'\d+-fold',
            r'top \d+%',
            r'ranked \#?\d+',
        ]
    }
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.QUANTIFICATION_PATTERNS.items()
        }
    
    def detect_action_verbs(self, text: str) -> Dict[str, Any]:
        """Detect action verbs in resume text.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dictionary with verb analysis
        """
        text_lower = text.lower()
        
        # Find strong action verbs
        strong_verbs_found = []
        for verb in self.ALL_STRONG_VERBS:
            # Look for verb at start of bullet or after common separators
            pattern = r'(?:^|[·•\-\*]|\n)\s*' + re.escape(verb.lower()) + r'\b'
            if re.search(pattern, text_lower, re.MULTILINE):
                strong_verbs_found.append(verb)
        
        # Find weak verbs using word boundary matching
        weak_verbs_found = []
        for verb in self.WEAK_VERBS:
            # Use word boundaries to avoid matching substrings (e.g., "made" in "Amadeus")
            pattern = r'\b' + re.escape(verb.lower()) + r'\b'
            if re.search(pattern, text_lower):
                weak_verbs_found.append(verb)
        
        # Categorize strong verbs
        verb_categories = {}
        for category, verbs in self.STRONG_ACTION_VERBS.items():
            category_verbs = [v for v in verbs if v in strong_verbs_found]
            if category_verbs:
                verb_categories[category] = category_verbs
        
        # Calculate score (0-25)
        bullet_count = max(len(self._extract_bullets(text)), 1)
        strong_verb_ratio = len(strong_verbs_found) / bullet_count
        
        # Score based on ratio (ideal: 70-100% of bullets start with strong verbs)
        if strong_verb_ratio >= 0.7:
            score = 25
        elif strong_verb_ratio >= 0.5:
            score = 20
        elif strong_verb_ratio >= 0.3:
            score = 15
        elif strong_verb_ratio >= 0.1:
            score = 10
        else:
            score = 5
        
        # Penalize weak verbs
        weak_verb_penalty = min(len(weak_verbs_found) * 2, 10)
        score = max(score - weak_verb_penalty, 0)
        
        return {
            'score': score,
            'strong_verbs': list(set(strong_verbs_found)),
            'weak_verbs': list(set(weak_verbs_found)),
            'categories': verb_categories,
            'coverage': strong_verb_ratio,
            'total_bullets': bullet_count,
            'strong_verb_count': len(strong_verbs_found),
        }
    
    def detect_quantification(self, text: str) -> Dict[str, Any]:
        """Detect quantified achievements in resume text.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dictionary with quantification analysis
        """
        quantified_items = []
        category_counts = {}
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check each category
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        quantified_items.append({
                            'text': line,
                            'category': category,
                            'match': match.group(0)
                        })
                        category_counts[category] = category_counts.get(category, 0) + 1
                        break  # Only count first match per line
        
        # Calculate score (0-25)
        bullet_count = max(len(self._extract_bullets(text)), 1)
        quantification_ratio = len(quantified_items) / bullet_count
        
        # Score based on ratio (ideal: 50%+ of bullets have metrics)
        if quantification_ratio >= 0.5:
            score = 25
        elif quantification_ratio >= 0.4:
            score = 20
        elif quantification_ratio >= 0.3:
            score = 15
        elif quantification_ratio >= 0.2:
            score = 10
        elif quantification_ratio >= 0.1:
            score = 5
        else:
            score = 0
        
        return {
            'score': score,
            'quantified_items': quantified_items,
            'category_counts': category_counts,
            'total_quantified': len(quantified_items),
            'coverage': quantification_ratio,
        }
    
    def analyze_bullets(self, text: str) -> Dict[str, Any]:
        """Analyze bullet point structure and readability.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dictionary with bullet analysis
        """
        bullets = self._extract_bullets(text)
        
        if not bullets:
            return {
                'score': 0,
                'bullet_count': 0,
                'avg_length': 0,
                'issues': ['No bullet points detected'],
                'bullets': [],
            }
        
        analyzed_bullets = []
        issues = []
        
        for bullet in bullets:
            word_count = len(bullet['text'].split())
            char_count = len(bullet['text'])
            
            # Determine bullet type and quality
            bullet_type = self._classify_bullet_type(bullet['text'])
            
            bullet_info = {
                'text': bullet['text'][:100] + '...' if len(bullet['text']) > 100 else bullet['text'],
                'word_count': word_count,
                'char_count': char_count,
                'type': bullet_type,
                'marker': bullet['marker'],
            }
            
            # Check for issues
            bullet_text_lower = bullet['text'].lower()
            issues_found = []
            
            # Length issues
            if word_count > 30:
                issues_found.append('Too long (ideal: 15-25 words)')
            elif word_count < 8:
                issues_found.append('Too short (may lack detail)')
            
            # Check for weak verbs at start of bullet
            starts_with_weak = False
            for weak_verb in self.WEAK_VERBS:
                pattern = r'^(?:\s*[·•\-\*]\s*)?' + re.escape(weak_verb.lower()) + r'\b'
                if re.match(pattern, bullet_text_lower):
                    starts_with_weak = True
                    issues_found.append(f'Starts with weak verb "{weak_verb}"')
                    break
            
            # Check for lack of quantification (if bullet seems achievement-oriented)
            has_quantification = False
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(bullet['text']):
                        has_quantification = True
                        break
                if has_quantification:
                    break
            
            if not has_quantification and word_count >= 8:
                # Check if this looks like an achievement bullet
                achievement_indicators = ['developed', 'created', 'built', 'implemented', 'launched', 
                                        'increased', 'decreased', 'improved', 'reduced', 'achieved',
                                        'delivered', 'designed', 'led', 'managed', 'spearheaded',
                                        'drove', 'generated', 'saved', 'grew', 'optimized']
                if any(indicator in bullet_text_lower for indicator in achievement_indicators):
                    issues_found.append('Missing metrics/quantification')
            
            # Check for passive voice indicators
            passive_patterns = [
                r'\bwas\s+\w+ed\b',
                r'\bwere\s+\w+ed\b',
                r'\bbeen\s+\w+ed\b',
                r'\bwas\s+responsible\s+for\b',
                r'\bwere\s+involved\s+in\b',
            ]
            for pattern in passive_patterns:
                if re.search(pattern, bullet_text_lower):
                    issues_found.append('Uses passive voice')
                    break
            
            # Check for filler phrases
            filler_phrases = [
                'in order to', 'due to the fact that', 'with regard to',
                'in the process of', 'at this point in time', 'for the purpose of',
                'in the event that', 'it is worth noting that'
            ]
            for filler in filler_phrases:
                if filler in bullet_text_lower:
                    issues_found.append('Contains filler phrases')
                    break
            
            if issues_found:
                bullet_info['issue'] = ' | '.join(issues_found)
            
            analyzed_bullets.append(bullet_info)
        
        # Calculate average length
        avg_words = sum(b['word_count'] for b in analyzed_bullets) / len(analyzed_bullets)
        
        # Score based on structure (0-25)
        score = 25
        
        # Penalize inconsistent lengths
        word_counts = [b['word_count'] for b in analyzed_bullets]
        if word_counts:
            max_count = max(word_counts)
            min_count = min(word_counts)
            if max_count - min_count > 20:
                score -= 5
                issues.append('Inconsistent bullet lengths')
        
        # Penalize too few bullets
        if len(bullets) < 3:
            score -= 5
            issues.append('Too few bullet points (aim for 3-5 per role)')
        
        # Penalize bullets that are too long on average
        if avg_words > 25:
            score -= 5
            issues.append('Bullets too verbose (aim for 15-25 words)')
        elif avg_words < 10:
            score -= 5
            issues.append('Bullets too brief (add more detail)')
        
        score = max(score, 0)
        
        return {
            'score': score,
            'bullet_count': len(bullets),
            'avg_length': avg_words,
            'issues': issues,
            'bullets': analyzed_bullets[:10],  # Limit to first 10
        }
    
    def analyze_conciseness(self, text: str) -> Dict[str, Any]:
        """Analyze text conciseness and readability.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dictionary with conciseness analysis
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Detect filler words and phrases
        filler_patterns = [
            r'\b(in order to)\b',
            r'\b(due to the fact that)\b',
            r'\b(at this point in time)\b',
            r'\b(for the purpose of)\b',
            r'\b(with regard to)\b',
            r'\b(in the event that)\b',
            r'\b(by means of)\b',
            r'\b(very|really|quite|rather|pretty)\s+\w+',
            r'\b(responsible for)\b',
            r'\b(duties include)\b',
        ]
        
        filler_count = 0
        filler_examples = []
        
        for line in lines:
            for pattern in filler_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                filler_count += len(matches)
                if matches and len(filler_examples) < 5:
                    filler_examples.append(line[:80])
        
        # Check for long paragraphs (vs bullets)
        paragraph_lines = [line for line in lines if len(line.split()) > 30 and not self._is_bullet_line(line)]
        
        # Calculate score (0-25)
        score = 25
        
        # Penalize filler words
        if filler_count > 5:
            score -= 10
        elif filler_count > 2:
            score -= 5
        
        # Penalize long paragraphs
        if len(paragraph_lines) > 3:
            score -= 10
        elif len(paragraph_lines) > 0:
            score -= 5
        
        score = max(score, 0)
        
        return {
            'score': score,
            'filler_count': filler_count,
            'filler_examples': filler_examples,
            'long_paragraphs': len(paragraph_lines),
            'issues': self._generate_conciseness_issues(filler_count, len(paragraph_lines)),
        }
    
    def analyze(self, text: str) -> ContentQualityScore:
        """Perform complete content quality analysis.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            ContentQualityScore with all metrics
        """
        # Run all analyses
        action_verb_analysis = self.detect_action_verbs(text)
        quantification_analysis = self.detect_quantification(text)
        bullet_analysis = self.analyze_bullets(text)
        conciseness_analysis = self.analyze_conciseness(text)
        
        # Calculate weighted overall score
        # Weights: Action verbs 30%, Quantification 30%, Bullets 20%, Conciseness 20%
        overall_score = int(
            action_verb_analysis['score'] * 0.30 +
            quantification_analysis['score'] * 0.30 +
            bullet_analysis['score'] * 0.20 +
            conciseness_analysis['score'] * 0.20
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            action_verb_analysis,
            quantification_analysis,
            bullet_analysis,
            conciseness_analysis
        )
        
        return ContentQualityScore(
            overall_score=overall_score,
            action_verb_score=action_verb_analysis['score'],
            quantification_score=quantification_analysis['score'],
            bullet_structure_score=bullet_analysis['score'],
            conciseness_score=conciseness_analysis['score'],
            action_verbs_found=action_verb_analysis['strong_verbs'],
            weak_verbs_found=action_verb_analysis['weak_verbs'],
            quantified_achievements=[item['text'][:100] for item in quantification_analysis['quantified_items'][:5]],
            bullet_points=bullet_analysis['bullets'],
            recommendations=recommendations,
        )
    
    def _extract_bullets(self, text: str) -> List[Dict[str, str]]:
        """Extract bullet points from text."""
        bullets = []
        lines = text.split('\n')
        
        bullet_markers = ['•', '·', '-', '*', '◦', '▪', '▫', '→', '⇒', '>', '○', '●']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for bullet markers
            for marker in bullet_markers:
                if line.startswith(marker):
                    bullets.append({
                        'text': line[len(marker):].strip(),
                        'marker': marker,
                    })
                    break
            else:
                # Check for numbered lists (1., 1), (a), etc.)
                if re.match(r'^[\d\w][\.\)]\s', line):
                    bullets.append({
                        'text': re.sub(r'^[\d\w][\.\)]\s', '', line),
                        'marker': line[0],
                    })
        
        return bullets
    
    def _is_bullet_line(self, line: str) -> bool:
        """Check if a line is a bullet point."""
        bullet_markers = ['•', '·', '-', '*', '◦', '▪', '▫', '→', '⇒', '>']
        line = line.strip()
        has_marker = any(line.startswith(marker) for marker in bullet_markers)
        is_numbered = re.match(r'^[\d\w][\.\)]\s', line) is not None
        return has_marker or is_numbered
    
    def _classify_bullet_type(self, text: str) -> str:
        """Classify the type of bullet point."""
        text_lower = text.lower()
        
        # Check for achievement patterns
        if any(pattern in text_lower for pattern in ['increased', 'decreased', 'improved', 'reduced', 'saved', 'grew']):
            return 'achievement'
        
        # Check for responsibility patterns
        if any(pattern in text_lower for pattern in ['responsible for', 'managed', 'led', 'oversaw']):
            return 'responsibility'
        
        # Check for skill/technology
        if any(pattern in text_lower for pattern in ['using', 'utilized', 'developed with', 'built with']):
            return 'technical'
        
        return 'general'
    
    def _generate_conciseness_issues(self, filler_count: int, paragraph_count: int) -> List[str]:
        """Generate conciseness issue descriptions."""
        issues = []
        
        if filler_count > 5:
            issues.append(f'{filler_count} filler phrases detected (remove unnecessary words)')
        elif filler_count > 2:
            issues.append(f'{filler_count} filler phrases detected')
        
        if paragraph_count > 3:
            issues.append(f'{paragraph_count} long paragraphs (convert to bullets)')
        elif paragraph_count > 0:
            issues.append('Some long paragraphs detected')
        
        return issues
    
    def _generate_recommendations(
        self,
        action_verb_analysis: Dict,
        quantification_analysis: Dict,
        bullet_analysis: Dict,
        conciseness_analysis: Dict
    ) -> List[str]:
        """Generate content improvement recommendations."""
        recommendations = []
        
        # Action verb recommendations
        if action_verb_analysis['score'] < 20:
            coverage = action_verb_analysis['coverage'] * 100
            recommendations.append(
                f"Start {100-coverage:.0f}% more bullets with strong action verbs "
                f"(led, developed, implemented, increased)"
            )
        
        if action_verb_analysis['weak_verbs']:
            recommendations.append(
                f"Replace weak verbs ({', '.join(action_verb_analysis['weak_verbs'][:3])}) "
                f"with stronger alternatives"
            )
        
        # Quantification recommendations
        if quantification_analysis['score'] < 15:
            recommendations.append(
                "Add metrics to achievements (%, $, time saved, users impacted)"
            )
        
        # Bullet structure recommendations
        if bullet_analysis['issues']:
            for issue in bullet_analysis['issues'][:2]:
                recommendations.append(issue)
        
        # Conciseness recommendations
        if conciseness_analysis['score'] < 20:
            if conciseness_analysis['filler_count'] > 0:
                recommendations.append("Remove filler words (very, really, in order to)")
            if conciseness_analysis['long_paragraphs'] > 0:
                recommendations.append("Convert paragraphs to bullet points")
        
        return recommendations[:5]  # Limit to top 5


# Convenience function
def analyze_content(text: str) -> ContentQualityScore:
    """Analyze resume content quality.
    
    Args:
        text: Resume text to analyze
        
    Returns:
        ContentQualityScore with analysis results
    """
    analyzer = ContentAnalyzer()
    return analyzer.analyze(text)
