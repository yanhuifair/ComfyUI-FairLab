# ComfyUI-FairLab

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-custom__nodes-blue?style=flat-square&logo=python" alt="ComfyUI Custom Nodes">
  <img src="https://img.shields.io/badge/version-1.0.90-green?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/nodes-58-orange?style=flat-square" alt="Nodes Count">
  <a href="https://github.com/yanhuifair/ComfyUI-FairLab/stargazers"><img src="https://img.shields.io/github/stars/yanhuifair/ComfyUI-FairLab?style=flat-square" alt="GitHub stars"></a>
</p>

<p align="center">
  A collection of 58 utility nodes for ComfyUI, spanning string processing, image I/O and manipulation,<br>
  arithmetic logic, PBR material tooling, and workflow diagnostics.<br>
  All nodes conform to the ComfyUI <code>IO.*</code> type annotation convention.
</p>

> Language: [中文文档](./README-zh.md)

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Method 1: ComfyUI Manager](#method-1-comfyui-manager-recommended)
  - [Method 2: Manual](#method-2-manual-installation)
  - [Verify Installation](#verify-installation)
  - [Updating](#updating)
  - [Dependencies](#dependencies)
  - [Troubleshooting](#troubleshooting)
- [Node Reference](#node-reference)
  - [String](#string-16-nodes)
  - [Image](#image-26-nodes)
  - [Logic](#logic-11-nodes)
  - [Utility](#utility-5-nodes)
- [Technical Notes](#technical-notes)
- [License](#license)

---

## Overview

ComfyUI-FairLab supplements the core ComfyUI node set with operations that are commonly needed in production workflows but not available in the base distribution.

| Category  | Nodes | Scope |
|:----------|:-----:|:------|
| String    | 16   | String construction, tag manipulation, translation, encoding repair, file I/O |
| Image     | 26   | File/URL/Base64 loading, batch I/O, video conversion, alpha & channel ops, PBR maps, modulation |
| Logic     | 11   | Arithmetic operators, conditional branching, type casting |
| Utility   | 5    | Diagnostic printing, aspect ratio presets, dual LoRA loading, sandboxed scripting |

---

## Installation

### Prerequisites

- **ComfyUI** installed and working. If not, follow the [official guide](https://github.com/comfyanonymous/ComfyUI#installing).
- **Python** 3.10+ (the same environment ComfyUI uses).
- **Git** (for manual installation).

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI and navigate to the **Manager** panel.
2. Click **Install Custom Nodes**.
3. Search for `ComfyUI-FairLab`.
4. Click **Install** and restart ComfyUI.

### Method 2: Manual Installation

```bash
# 1. Navigate to the custom_nodes directory
cd ComfyUI/custom_nodes

# 2. Clone the repository
git clone https://github.com/yanhuifair/ComfyUI-FairLab.git

# 3. Enter the project directory
cd ComfyUI-FairLab

# 4. Install dependencies
pip install -r requirements.txt

# 5. Restart ComfyUI
```

### Verify Installation

After restarting ComfyUI, double-click the canvas and search for `FairLab` or any node name (e.g., `String Append`, `Load Image Batch From Directory`). If the nodes appear in the search results, the installation was successful.

### Updating

**ComfyUI Manager:** Click **Update All** in the Manager panel.

**Manual:**
```bash
cd ComfyUI/custom_nodes/ComfyUI-FairLab
git pull
pip install -r requirements.txt
```

### Dependencies

| Package | Required By |
|:--------|:------------|
| `googletrans`                      | String Translate              |
| `opencv-python`                    | Video ↔ image; general processing |
| `requests`                         | Download Image, Load Image From URL |
| `nest_asyncio`                     | Async event loop support      |
| `perfect-pixel[opencv]>=0.1.4`     | Perfect Pixel                 |

### Troubleshooting

**Nodes not showing up after installation**

- Make sure you restarted ComfyUI after installing.
- Check the ComfyUI terminal for import errors related to `ComfyUI-FairLab`.
- Verify that dependencies were installed: `pip list | grep -E "opencv|googletrans|perfect-pixel"`

**ImportError: No module named 'cv2'**

```bash
pip install opencv-python
```

**Google Translate not working**

The `googletrans` library may require an updated version. Try:
```bash
pip install --upgrade googletrans
```

---

## Node Reference

### String (16 nodes)

String construction, persistence, and tag-oriented manipulation.

| Node | Description |
|:-----|:------------|
| **String**                 | Emit a constant string value |
| **Int**                    | Emit a constant integer value |
| **Float**                  | Emit a constant floating-point value |
| **String Append**          | Concatenate multiple input strings into a single output |
| **Load String**            | Read string content from a local text file |
| **Save String To Directory**   | Write a string to a file on disk |
| **Load String From Directory** | Read a string from a file in a specified directory |
| **Show String**            | Display a string value on the node's UI panel for inspection |
| **String Translate**       | Translate strings via Google Translate (source/target language configurable) |
| **Fix UTF-8 String**       | Repair malformed UTF-8 byte sequences (e.g., mojibake from encoding mismatches) |
| **Range String**           | Generate a sequence of indexed strings over a numeric range (e.g., `frame_001`..`frame_100`) |
| **Prepend Tags**           | Insert a prefix before each tag in a comma-separated list |
| **Append Tags**            | Append a suffix to each tag in a comma-separated list |
| **Exclude Tags**           | Remove specified tags from a tag list |
| **Unique Tags**            | Deduplicate entries in a tag list |
| **ASCII Art Text**         | Render text as an ASCII-art image using system fonts |

### Image (26 nodes)

#### I/O & Loading

| Node | Description |
|:-----|:------------|
| **Load Image From Directory**       | Load a single image from a directory |
| **Load Image Batch From Directory** | Load all images from a directory as a batch tensor |
| **Load Image From URL**             | Fetch an image directly from a remote URL into a tensor |
| **Download Image**                  | Download a remote image and persist it to local disk |
| **Save Image To Directory**         | Write an image tensor to a specified output path |
| **Save Image To Folder**            | Save images organized by subdirectory |
| **Image To Base64**                 | Encode an image tensor as a Base64 string |
| **Base64 To Image**                 | Decode a Base64 string into an image tensor |

#### Processing

| Node | Description |
|:-----|:------------|
| **Resize Image**           | Resize an image tensor to target dimensions |
| **Image Size**             | Output the width and height of an image as scalar values |
| **Image Shape**            | Output the full tensor shape `[B, C, H, W]` of an image batch |
| **Image Remove Alpha**     | Strip the alpha channel from an RGBA image, producing RGB |
| **Fill Alpha**             | Add or replace the alpha channel on an RGB image |
| **Pure Color Image**       | Generate a solid-color image of configurable dimensions and RGB value |
| **Images Range**           | Slice a sub-range of images from a batch by start/end index |
| **Images Index**           | Extract a single image from a batch by index |
| **Images Cat**             | Concatenate multiple image batches along the batch dimension |
| **Outpainting Pad**        | Pad image borders for outpainting workflows |
| **Perfect Pixel**          | Integer-scale an image without sub-pixel interpolation (nearest-neighbor at exact multiples) |

#### Video

| Node | Description |
|:-----|:------------|
| **Video To Image**   | Extract frames from a video file as an image batch tensor |
| **Image To Video**   | Compose an image batch into a video file (configurable FPS) |

#### Modulation

| Node | Description |
|:-----|:------------|
| **Modulation**            | Apply error-diffusion modulation (Floyd–Steinberg-style halftoning) |
| **Modulation Direction**  | Directional error-diffusion with configurable scan direction and speed |

#### PBR Maps

| Node | Description |
|:-----|:------------|
| **Mask Map**                  | Convert a grayscale image into a normalized mask tensor |
| **Detail Map**                | Generate a detail map from an input image for PBR material pipelines |
| **Roughness To Smoothness**   | Convert between roughness and smoothness maps (invert) |

### Logic (11 nodes)

Arithmetic and control-flow primitives for node-graph-level computation.

| Node | Description |
|:-----|:------------|
| **Number**         | Generic numeric constant (INT / FLOAT toggle) |
| **Add**            | `A + B` with automatic INT/FLOAT type resolution |
| **Subtract**       | `A - B` |
| **Multiply**       | `A × B` with automatic type resolution |
| **Multiply Int**   | `A × B` with result cast to INT |
| **Divide**         | `A ÷ B` |
| **Max**            | `max(A, B)` |
| **Min**            | `min(A, B)` |
| **If**             | Conditional branch: route to output A or B based on a boolean condition |
| **Float To Int**   | Cast FLOAT to INT (truncation) |
| **Int To Float**   | Cast INT to FLOAT |

### Utility (5 nodes)

Diagnostics, presets, and extensibility.

| Node | Description |
|:-----|:------------|
| **Print Any**        | Print any input value to the console for debugging |
| **Print Image**      | Print image tensor metadata to the console (shape, dtype, value range) |
| **Aspect Ratios**    | Dropdown selector for common aspect ratios (1:1, 16:9, 4:3, 3:2, 2:3, 21:9) |
| **Load LoRA Dual**   | Load two LoRA models simultaneously with independent strength parameters |
| **Python Script**    | Evaluate a restricted Python expression in an AST-whitelist sandbox; supports arithmetic, comparison operators, and a curated set of builtins (`abs`, `min`, `max`, `round`, `len`, etc.). `exec` and `eval` are explicitly disabled. |

---

## Technical Notes

**Python Script sandbox**

The expression evaluator uses an AST whitelist that permits only explicitly registered operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`, `<`, `>`, `<=`, `>=`, `==`, `!=`, `is`, `is not`, `not`, unary `+`/`-`) and built-in functions (`abs`, `bool`, `float`, `int`, `len`, `max`, `min`, `round`, `sorted`, `str`, `sum`, `tuple`, `list`). Arbitrary code execution via `exec` or `eval` is not possible.

**IO typing**

All nodes use the ComfyUI `IO.*` type annotations (`IO.STRING`, `IO.IMAGE`, `IO.INT`, `IO.FLOAT`, etc.), ensuring compatibility with the modern ComfyUI type system.

**ASCII Art Text**

Requires system font support. On minimal Linux installations, install a font package (e.g., `fonts-dejavu-core` on Debian/Ubuntu).

**Video operations**

Frame extraction and composition rely on `opencv-python`. Processing time scales with resolution and frame count.

---

## License

MIT — see [LICENSE](./LICENSE).

Copyright (c) 2025 [Fair](https://github.com/yanhuifair)
