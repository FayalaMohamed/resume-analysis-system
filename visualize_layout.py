#!/usr/bin/env python3
"""
Visualize ML Layout Detection Regions

This script visualizes what the ML layout detection model sees in a resume.
It draws colored bounding boxes around detected regions (text, titles, tables, figures, etc.)

Usage:
    python visualize_layout.py resume.pdf [--output-dir visualizations/]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers.ml_layout_detector import MLLayoutDetector

# Color scheme for different region types
REGION_COLORS = {
    'text': '#3498db',      # Blue
    'title': '#e74c3c',     # Red
    'figure': '#2ecc71',    # Green
    'image': '#2ecc71',     # Green (same as figure)
    'table': '#f39c12',     # Orange
    'list': '#9b59b6',      # Purple
    'reference': '#1abc9c', # Teal
    'unknown': '#95a5a6',   # Gray
}

# Font settings for labels
LABEL_FONT_SIZE = 16
LABEL_PADDING = 4


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> List[Path]:
    """Convert PDF pages to images."""
    doc = fitz.open(str(pdf_path))
    temp_dir = pdf_path.parent / pdf_path.stem / ".temp_vis"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Use matrix for DPI conversion
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        image_path = temp_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)
    
    doc.close()
    return image_paths


def draw_regions_on_image(image_path: Path, regions: List[Dict], output_path: Path):
    """Draw bounding boxes around detected regions."""
    # Open image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", LABEL_FONT_SIZE)
        except:
            font = ImageFont.load_default()
    
    # Sort regions by confidence (highest first) so most confident are drawn on top
    sorted_regions = sorted(regions, key=lambda r: r.get('confidence', 0), reverse=True)
    
    # Debug: print first few regions to see bbox format
    if sorted_regions:
        print(f"\n     Debug - Sample region: {sorted_regions[0]}")
    
    drawn_count = 0
    skipped_count = 0
    
    # Draw each region
    for i, region in enumerate(sorted_regions):
        region_type = region.get('type', 'unknown')
        bbox = region.get('bbox', [])
        confidence = region.get('confidence', 0)
        
        # Debug: show bbox format for first few regions
        if i < 3:
            print(f"     Region {i+1}: type={region_type}, bbox={bbox}, conf={confidence:.2f}")
        
        if not bbox or len(bbox) < 4:
            skipped_count += 1
            if i < 3:
                print(f"       -> Skipped: invalid bbox length")
            continue
        
        # Get color for this region type
        color = REGION_COLORS.get(region_type, REGION_COLORS['unknown'])
        
        # Draw bounding box - convert to integers and ensure proper format
        try:
            # Handle different bbox formats
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                # Check if bbox might be nested (e.g., [[x1,y1,x2,y2]])
                if len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
                    bbox = bbox[0]
                
                # Handle format variations
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [float(coord) for coord in bbox[:4]]
                elif len(bbox) == 2 and all(len(b) == 2 for b in bbox):
                    # Format: [[x1,y1], [x2,y2]]
                    (x1, y1), (x2, y2) = bbox
                else:
                    x1, y1, x2, y2 = [float(coord) for coord in bbox[:4]]
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Check if box has valid size
                if x2 - x1 < 5 or y2 - y1 < 5:
                    skipped_count += 1
                    if i < 3:
                        print(f"       -> Skipped: box too small ({x2-x1}x{y2-y1})")
                    continue
                
                # Draw the bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                drawn_count += 1
                
                if i < 3:
                    print(f"       -> Drew box at ({x1},{y1}) to ({x2},{y2})")
                
                # Create label text
                label = f"{region_type.upper()} ({confidence:.0%})"
                
                # Calculate label position and size
                bbox_label = draw.textbbox((0, 0), label, font=font)
                label_width = bbox_label[2] - bbox_label[0]
                label_height = bbox_label[3] - bbox_label[1]
                
                # Position label above the box (or below if at top of page)
                label_x = x1
                label_y = y1 - label_height - LABEL_PADDING * 2
                if label_y < 0:  # If too close to top, put it inside the box at top
                    label_y = y1 + LABEL_PADDING
                
                # Draw label background
                bg_bbox = [
                    label_x,
                    label_y,
                    label_x + label_width + LABEL_PADDING * 2,
                    label_y + label_height + LABEL_PADDING * 2
                ]
                draw.rectangle(bg_bbox, fill=color)
                
                # Draw label text
                draw.text(
                    (label_x + LABEL_PADDING, label_y + LABEL_PADDING),
                    label,
                    fill='white',
                    font=font
                )
                
                # Draw region number for reference
                num_label = str(i + 1)
                draw.text(
                    (x1 + 5, y2 - 20),
                    num_label,
                    fill=color,
                    font=font
                )
            else:
                skipped_count += 1
                print(f"    Skipping region {i+1}: invalid bbox format {bbox}")
                continue
        except (ValueError, TypeError, IndexError) as e:
            skipped_count += 1
            print(f"    Skipping region {i+1}: could not parse bbox {bbox} - {e}")
            continue
    
    print(f"\n     Summary: {drawn_count} boxes drawn, {skipped_count} skipped")
    
    # Add legend
    legend_y = 10
    legend_x = img.width - 200
    
    # Draw legend background
    legend_height = len(REGION_COLORS) * 25 + 20
    draw.rectangle(
        [legend_x - 10, legend_y - 10, legend_x + 190, legend_y + legend_height],
        fill='white',
        outline='black',
        width=2
    )
    
    # Draw legend title
    draw.text((legend_x, legend_y), "REGION TYPES:", fill='black', font=font)
    legend_y += 25
    
    # Draw legend items
    for region_type, color in REGION_COLORS.items():
        # Color box
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 15], fill=color)
        # Label
        draw.text((legend_x + 25, legend_y), region_type.upper(), fill='black', font=font)
        legend_y += 25
    
    # Add summary text
    summary_y = 10
    summary_x = 10
    total_regions = len(sorted_regions)
    
    draw.rectangle(
        [summary_x - 5, summary_y - 5, summary_x + 300, summary_y + 100],
        fill='white',
        outline='black',
        width=2
    )
    
    draw.text((summary_x, summary_y), "DETECTION SUMMARY:", fill='black', font=font)
    summary_y += 25
    draw.text((summary_x, summary_y), f"Total Regions: {total_regions}", fill='black', font=font)
    summary_y += 25
    
    # Count by type
    type_counts = {}
    for r in sorted_regions:
        t = r.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for region_type, count in type_counts.items():
        draw.text((summary_x, summary_y), f"  {region_type}: {count}", fill='black', font=font)
        summary_y += 20
    
    # Save the visualized image
    img.save(output_path)
    print(f"  ✓ Saved visualization: {output_path}")


def visualize_resume_layout(pdf_path: Path, output_dir_input = None):
    """Main function to visualize layout detection on a resume."""
    print(f"\n{'='*70}")
    print(f"LAYOUT VISUALIZATION: {pdf_path.name}")
    print(f"{'='*70}\n")
    
    # Setup output directory
    if output_dir_input is None:
        output_dir = pdf_path.parent / pdf_path.stem / "layout_visualization"
    else:
        output_dir = Path(output_dir_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Check if ML detector is available
    detector = MLLayoutDetector()
    if not detector.is_available():
        print("\n[ERROR] ML LayoutDetection is not available!")
        print("  Please ensure PaddleOCR is installed with LayoutDetection support.")
        return
    
    print("\n1. Converting PDF to images...")
    image_paths = pdf_to_images(pdf_path)
    print(f"   ✓ Created {len(image_paths)} page images")
    
    print("\n2. Running ML layout detection...")
    all_regions = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n   Page {i + 1}:")
        
        # Detect regions
        regions = detector.detect_layout(str(image_path))
        all_regions.extend(regions)
        
        print(f"     Detected {len(regions)} regions:")
        
        # Count by type
        type_counts = {}
        for r in regions:
            t = r.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        
        for region_type, count in type_counts.items():
            print(f"       - {region_type}: {count}")
        
        # Warn if all regions are unknown
        if len(regions) > 0 and all(r.get('type') == 'unknown' for r in regions):
            print(f"     ⚠ WARNING: All regions detected as 'unknown'")
            print(f"       This may indicate the model didn't recognize the content properly.")
        
        # Create visualization
        output_image_path = output_dir / f"{pdf_path.stem}_page_{i + 1}_visualized.png"
        draw_regions_on_image(image_path, regions, output_image_path)
    
    # Print overall summary
    print(f"\n{'='*70}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal regions detected across all pages: {len(all_regions)}")
    print(f"\nOutput files saved to: {output_dir}")
    
    # List all output files
    output_files = list(output_dir.glob("*_visualized.png"))
    print(f"\nGenerated files:")
    for f in output_files:
        print(f"  - {f.name}")
    
    # Cleanup temp images
    temp_dir = pdf_path.parent / pdf_path.stem / ".temp_vis"
    if temp_dir.exists():
        for img_path in temp_dir.glob("*.png"):
            img_path.unlink()
        temp_dir.rmdir()
    
    print("\n✓ Done! Open the visualized images to see what the ML model detected.")
    print(f"  Tip: Look for colored boxes - each color represents a different region type.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ML layout detection regions on resume PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_layout.py resume.pdf
  python visualize_layout.py resume.pdf --output-dir my_visualizations/
  
The script will create visualized images showing colored boxes around
detected regions (text, titles, tables, figures, etc.)

Color Legend:
  Blue   = Text
  Red    = Title
  Green  = Figure/Image
  Orange = Table
  Purple = List
  Teal   = Reference
  Gray   = Unknown
        """
    )
    
    parser.add_argument("resume", help="Path to resume PDF file")
    parser.add_argument("--output-dir", "-o", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    visualize_resume_layout(resume_path, output_dir)


if __name__ == "__main__":
    main()
