# Web Skeleton Viewer Design

## Overview

A lightweight web-based 3D viewer for MHR70 skeleton keypoints, designed to embed into existing plain HTML/JS web projects.

## Components

### 1. JSON Exporter (`tools/json_export.py`)

Python module that exports skeleton keypoints to a web-friendly JSON format.

**JSON Format:**
```json
{
  "version": "1.0",
  "source_image": "18853875.jpg",
  "keypoints": [
    [0.123, 1.456, 0.789],
    ...
  ],
  "keypoint_names": ["nose", "left_eye", "right_eye", ...],
  "bones": [
    [13, 11], [11, 9], [14, 12], ...
  ],
  "bone_colors": [
    [0, 255, 0], [0, 255, 0], [255, 128, 0], ...
  ]
}
```

- **keypoints**: Array of 70 `[x, y, z]` coordinates
- **keypoint_names**: Joint names for hover tooltips
- **bones**: Index pairs defining connections (from MHR70 skeleton_info)
- **bone_colors**: RGB values (left=green, right=orange, center=blue)

### 2. Skeleton Viewer (`viewer/skeleton_viewer.html`)

Self-contained HTML file with embedded CSS and JavaScript.

**Dependencies (CDN):**
- Three.js r150+
- OrbitControls

**Structure:**
```
skeleton_viewer.html
├── <style>           - Dark theme, fullscreen canvas
├── <script> Three.js - From CDN
├── <script> OrbitControls - From CDN
└── <script> App code
    ├── init()        - Scene, camera, renderer setup
    ├── loadSkeleton(url) - Fetch JSON, build geometry
    ├── buildBones()  - Line segments for skeleton
    ├── buildJoints() - Spheres at keypoints
    └── animate()     - Render loop
```

**Visual Design:**
- Background: Dark gray (#1a1a1a)
- Joints: White spheres (radius ~0.02)
- Bones: Colored lines from bone_colors
- Ground: Subtle grid for reference

**Loading Methods:**
1. URL parameter: `?file=path/to/skeleton.json`
2. Drag-and-drop JSON onto canvas

**Controls:**
- Left-drag: Rotate
- Scroll: Zoom
- Right-drag: Pan

## Data Flow

```
Image → SAM 3D Body → JSON export → skeleton_viewer.html → Interactive 3D view
```

## File Structure

```
sam-3d-body/
├── tools/
│   └── json_export.py      # New: JSON export function
├── viewer/
│   └── skeleton_viewer.html # New: Self-contained viewer
└── output/
    └── samples/
        └── *.json          # Exported skeleton files
```

## Future Extensions

- Multi-person support (show all detected skeletons)
- Animation playback for sequences
- Keypoint labels on hover
