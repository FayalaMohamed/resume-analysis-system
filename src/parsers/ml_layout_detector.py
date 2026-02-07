"""ML-based layout detection using PaddleOCR LayoutDetection."""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Check LayoutDetection availability (PaddleOCR 3.x API)
try:
    from paddleocr import LayoutDetection as PaddleLayoutDetection
    LAYOUT_DETECTION_AVAILABLE = True
except ImportError:
    LAYOUT_DETECTION_AVAILABLE = False
    PaddleLayoutDetection = None
    logger.debug("PaddleOCR LayoutDetection not available, will use heuristic detection")


class MLLayoutDetector:
    """ML-based layout detection using PaddleOCR LayoutDetection."""
    
    TABLE_CONFIDENCE_THRESHOLD = 0.7
    COLUMN_GAP_THRESHOLD = 50  # pixels
    IMAGE_DPI = 150
    
    def __init__(self, lang: str = 'en'):
        """Initialize ML layout detector.
        
        Args:
            lang: Language code (not used in new API but kept for compatibility)
        """
        self._detector = None  # Lazy loading
        self._available = None
        self.lang = lang
    
    def is_available(self) -> bool:
        """Check if LayoutDetection can be loaded."""
        if self._available is not None:
            return self._available
        
        if not LAYOUT_DETECTION_AVAILABLE:
            self._available = False
            return False
        
        try:
            self._load_model()
            self._available = True
        except Exception as e:
            logger.warning(f"LayoutDetection not available: {e}")
            self._available = False
        
        return self._available
    
    def _load_model(self):
        """Load LayoutDetection model on first use."""
        if self._detector is not None:
            return
        
        logger.info("Loading PaddleOCR LayoutDetection model...")
        self._detector = PaddleLayoutDetection()
        logger.info("LayoutDetection model loaded successfully")
    
    def detect_layout(self, image_path: str) -> List[Dict]:
        """Run LayoutDetection inference on a page image.
        
        Args:
            image_path: Path to the page image
            
        Returns:
            List of detected regions with type, bbox, and confidence
        """
        if not self.is_available():
            raise RuntimeError("LayoutDetection is not available")
        
        result = self._detector.predict(image_path)
        
        regions = []
        # Handle PaddleOCR 3.x result format
        for res in result:
            try:
                # The result is a DetResult object that acts like a dict
                # Access boxes directly from the result dict
                if 'boxes' in res:
                    boxes = res['boxes']
                    logger.debug(f"Found {len(boxes)} boxes in result")
                    
                    for box in boxes:
                        # Each box is a dict with: cls_id, label, score, coordinate
                        label = box.get('label', 'unknown')
                        score = box.get('score', 0.0)
                        coordinate = box.get('coordinate', [])
                        
                        # Convert coordinate (list of np.float32) to regular floats
                        if coordinate and len(coordinate) >= 4:
                            bbox_coords = [float(coordinate[i]) for i in range(4)]
                        else:
                            bbox_coords = []
                        
                        region = {
                            'type': self._normalize_label(label),
                            'bbox': bbox_coords,
                            'confidence': float(score) if score else 0.0,
                        }
                        regions.append(region)
                        
            except Exception as e:
                logger.warning(f"Error parsing detection result: {e}")
                continue
        
        return regions
    
    def _normalize_label(self, label: Any) -> str:
        """Normalize layout labels to standard types."""
        if isinstance(label, (int, float)):
            # Map numeric labels to types (based on PP-DocLayout)
            label_map = {
                0: 'text',
                1: 'title',
                2: 'figure',
                3: 'table',
                4: 'list',
                5: 'reference',
            }
            return label_map.get(int(label), 'unknown')
        
        label_str = str(label).lower()
        if 'text' in label_str or 'paragraph' in label_str:
            return 'text'
        elif 'title' in label_str or 'header' in label_str:
            return 'title'
        elif 'table' in label_str:
            return 'table'
        elif 'figure' in label_str or 'image' in label_str:
            return 'figure'
        elif 'list' in label_str:
            return 'list'
        return label_str
    
    def analyze_columns(self, regions: List[Dict], page_width: float = 612) -> Dict[str, Any]:
        """Analyze text regions to detect column layout.
        
        Args:
            regions: List of detected regions from LayoutDetection
            page_width: Width of the page in pixels
            
        Returns:
            Column detection results
        """
        # Filter text and title regions
        text_regions = [
            r for r in regions 
            if r.get('type') in ('text', 'title', 'reference')
        ]
        
        if not text_regions:
            return {
                'num_columns': 1,
                'is_single_column': True,
                'confidence': 0.5,
                'column_regions': []
            }
        
        # Sort by y-coordinate (top of bbox)
        sorted_regions = sorted(
            text_regions, 
            key=lambda r: r['bbox'][1] if len(r['bbox']) >= 4 else 0
        )
        
        # Analyze x-positions to detect columns
        x_positions = []
        for r in sorted_regions:
            if len(r['bbox']) >= 4:
                # Use center x position
                x_center = (r['bbox'][0] + r['bbox'][2]) / 2
                x_positions.append(x_center)
        
        if not x_positions:
            return {
                'num_columns': 1,
                'is_single_column': True,
                'confidence': 0.5,
                'column_regions': []
            }
        
        # Cluster x-positions to detect columns
        mid_page = page_width / 2
        
        left_count = sum(1 for x in x_positions if x < mid_page - self.COLUMN_GAP_THRESHOLD)
        right_count = sum(1 for x in x_positions if x > mid_page + self.COLUMN_GAP_THRESHOLD)
        
        # If both sides have significant content, it's multi-column
        total = len(x_positions)
        left_ratio = left_count / total if total > 0 else 0
        right_ratio = right_count / total if total > 0 else 0
        
        is_multi_column = left_ratio > 0.2 and right_ratio > 0.2
        num_columns = 2 if is_multi_column else 1
        
        # Calculate confidence based on distribution clarity
        confidence = max(left_ratio, right_ratio) if is_multi_column else 0.9
        
        return {
            'num_columns': num_columns,
            'is_single_column': num_columns == 1,
            'confidence': confidence,
            'column_regions': text_regions
        }
    
    def analyze_tables(self, regions: List[Dict]) -> Dict[str, Any]:
        """Extract table regions with confidence.
        
        Args:
            regions: List of detected regions from LayoutDetection
            
        Returns:
            Table detection results
        """
        table_regions = [
            r for r in regions 
            if r.get('type') == 'table' and r.get('confidence', 0) >= self.TABLE_CONFIDENCE_THRESHOLD
        ]
        
        has_tables = len(table_regions) > 0
        avg_confidence = (
            sum(r.get('confidence', 0) for r in table_regions) / len(table_regions)
            if table_regions else 0.0
        )
        
        return {
            'has_tables': has_tables,
            'table_count': len(table_regions),
            'confidence': avg_confidence,
            'table_regions': table_regions
        }
    
    def analyze_images(self, regions: List[Dict]) -> Dict[str, Any]:
        """Detect image/figure regions.
        
        Args:
            regions: List of detected regions from LayoutDetection
            
        Returns:
            Image detection results
        """
        image_regions = [
            r for r in regions 
            if r.get('type') in ('figure', 'image')
        ]
        
        return {
            'has_images': len(image_regions) > 0,
            'image_count': len(image_regions),
            'image_regions': image_regions
        }
