"""Layout detection types and data structures."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LayoutFeatures:
    """Features extracted from resume layout analysis."""
    is_single_column: bool
    has_tables: bool
    has_images: bool
    num_columns: int
    text_density: float
    avg_line_length: float
    section_headers: List[str]
    layout_risk_score: float
    # Fields for ML detection
    detection_method: str = "heuristic"  # "ml" or "heuristic"
    confidence: Optional[float] = None
    table_regions: Optional[List[Dict]] = field(default=None)
    column_regions: Optional[List[Dict]] = field(default=None)
