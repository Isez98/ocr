#!/usr/bin/env python3
"""
Command-line tool to apply snap-to-ink refinement to existing templates.
"""

import argparse
import json
import os
import sys
from snap_to_ink import apply_snap_to_ink_interactive, snap_template_to_ink


def main():
    parser = argparse.ArgumentParser(
        description="Apply snap-to-ink auto-refinement to OCR templates"
    )
    parser.add_argument(
        "template_path", 
        help="Path to the template JSON file"
    )
    parser.add_argument(
        "--image-path", 
        help="Path to canonical image (if different from template)"
    )
    parser.add_argument(
        "--auto-apply", 
        action="store_true",
        help="Automatically apply changes without preview"
    )
    parser.add_argument(
        "--output", 
        help="Output path for refined template (default: overwrites input)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.template_path):
        print(f"Error: Template file not found: {args.template_path}")
        sys.exit(1)
    
    # Load template
    try:
        with open(args.template_path, 'r') as f:
            template = json.load(f)
    except Exception as e:
        print(f"Error loading template: {e}")
        sys.exit(1)
    
    # Determine image path
    if args.image_path:
        image_path = args.image_path
    else:
        # Try to find canonical image from template
        canonical_rel = template.get("canonical_image", "")
        if canonical_rel:
            # Try relative to template directory
            template_dir = os.path.dirname(args.template_path)
            image_path = os.path.join(template_dir, os.path.basename(canonical_rel))
            
            if not os.path.exists(image_path):
                # Try relative to current directory
                image_path = canonical_rel
                
            if not os.path.exists(image_path):
                print(f"Error: Cannot find canonical image: {canonical_rel}")
                print("Please specify --image-path manually")
                sys.exit(1)
        else:
            print("Error: No canonical image specified in template")
            print("Please specify --image-path manually")
            sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"Template: {args.template_path}")
    print(f"Image: {image_path}")
    print(f"ROIs: {len(template.get('rois', []))}")
    
    if args.auto_apply:
        # Auto-apply without preview
        try:
            refined_template, _ = snap_template_to_ink(image_path, template, preview_mode=False)
            
            # Save output
            output_path = args.output or args.template_path
            
            if output_path == args.template_path:
                # Backup original
                backup_path = args.template_path + '.backup'
                with open(backup_path, 'w') as f:
                    json.dump(template, f, indent=2)
                print(f"Original backed up to: {backup_path}")
            
            with open(output_path, 'w') as f:
                json.dump(refined_template, f, indent=2)
                
            print(f"Refined template saved to: {output_path}")
            
            # Show summary of changes
            changes = 0
            for original, refined in zip(template["rois"], refined_template["rois"]):
                if (original['x'] != refined['x'] or original['y'] != refined['y'] or
                    original['w'] != refined['w'] or original['h'] != refined['h']):
                    changes += 1
            
            print(f"Changes applied to {changes} out of {len(template['rois'])} ROIs")
            
        except Exception as e:
            print(f"Error processing template: {e}")
            sys.exit(1)
    else:
        # Interactive mode with preview
        apply_snap_to_ink_interactive(image_path, args.template_path)


if __name__ == "__main__":
    main()