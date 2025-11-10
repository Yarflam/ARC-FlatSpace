# Pixel Relative Knowledge (PRK) System for ARC Prize

**Historical Approach - PIXMELT Research**

---

## Overview

The Pixel Relative Knowledge (PRK) System introduced a cellular automaton-inspired approach that analyzed local pixel environments to determine transformation priorities. This classical algorithm produces reproducible results through deterministic rule application.

## Core Algorithm

```python
def get_pixel_attention_optimized(
    inputs: torch.Tensor,  # shape: [?, 30, 30], integers [0;10]
    markers: torch.Tensor,  # shape: [?, 30, 30], floatting [0;1]
    targets: torch.Tensor,  # shape: [?, 30, 30], integers [0;10]
    fill: float = 10.0
):
    states = torch.arange(0, 5, dtype=torch.float32, device=inputs.device) / 4.0
    batch_size, height, width = inputs.shape
    
    # Padding to handle borders
    inputs_pad = F.pad(inputs.float(), (1, 1, 1, 1), value=fill)
    markers_pad = F.pad(markers, (1, 1, 1, 1), value=0.0)
    
    # Extract 3x3 neighborhoods vectorized
    inputs_near = inputs_pad.unfold(1, 3, 1).unfold(2, 3, 1)  # [batch, H, W, 3, 3]
    markers_near = markers_pad.unfold(1, 3, 1).unfold(2, 3, 1)  # [batch, H, W, 3, 3]
    
    # Reshape for easier processing
    inputs_flat = inputs_near.reshape(batch_size, height, width, 9)
    markers_flat = markers_near.reshape(batch_size, height, width, 9)
    
    # Initialize result with state[4] (default case)
    result = torch.full_like(inputs, states[4], dtype=torch.float32)
    
    # Calculate pixel_match for all pixels\n    pixel_match = (targets != inputs)
    
    # Apply rules for input_diversity and marker_diversity\n    # Take the maximum of both rule results for final state determination\n    \n    return result, inputs_flat, markers_flat
```

## Framework Integration

The PRK system operates within a broader framework using step-by-step analysis and deterministic pattern labeling:

```python
def pattern_to_seed(pattern):
    pattern_bytes = pattern.cpu().numpy().tobytes()
    hash_obj = hashlib.md5(pattern_bytes)
    seed = int(hash_obj.hexdigest()[:8], 16)
    return seed

def get_vcnn_steps(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_steps: int = 1000,
    num_classes: int = 11,
    max_patterns: int = 200,
    echo: callable = print
):
    # Step-by-step transformation analysis with attention mechanisms
    # Pattern identification and marker propagation system
    # Deterministic rule application for reproducible results
    return steps, top_size
```

## PRK State Classification System

- **State 0.0**: No change + low diversity (< 2 unique values)
- **State 1.0**: No change + high diversity (≥ 2 unique values)  
- **State 2.0**: Has change + center pixel unchanged
- **State 3.0**: Has change + center changed + low diversity
- **State 4.0**: Has change + center changed + high diversity

## Architecture Design

The PRK system operated on a multi-layered representation:
- X binary one-hot layers (10 classes)
- Annotation layer for change propagation
- PRK analysis layer

Multiple expert modules (simple linear transformations using CNN2D structure) sequentially processed these layers to propagate changes from "hot zones" (high PRK diversity areas).

## Demonstration Results

### Version 1 - Initial Direct Drawing Approach

**Puzzle ID: 09c534e7**

<img src="demos/prk_examples/v1-09c534e7-ex0.gif" width="250" alt="PRK v1 Example 0">
<img src="demos/prk_examples/v1-09c534e7-ex1.gif" width="250" alt="PRK v1 Example 1">
<img src="demos/prk_examples/v1-09c534e7-ex2.gif" width="250" alt="PRK v1 Example 2">

*Initial cellular automaton-based transformation showing hot zone identification without propagation system.*

### Version 2 - First Propagation Implementation

**Puzzle ID: 007bbfb7**  
<img src="demos/prk_examples/v2-007bbfb7-ex1.gif" width="250" alt="PRK v2 - 007bbfb7 ex1">

**Puzzle ID: 00d62c1b**  
<img src="demos/prk_examples/v2-00d62c1b-ex0.gif" width="250" alt="PRK v2 Example 0">
<img src="demos/prk_examples/v2-00d62c1b-ex1.gif" width="250" alt="PRK v2 Example 1">
<img src="demos/prk_examples/v2-00d62c1b-ex2.gif" width="250" alt="PRK v2 Example 2">
<img src="demos/prk_examples/v2-00d62c1b-ex3.gif" width="250" alt="PRK v2 Example 3">
<img src="demos/prk_examples/v2-00d62c1b-ex4.gif" width="250" alt="PRK v2 Example 4">

*Step-by-step evolution showing first propagation mechanisms with scaling issues and localized losses across different image regions.*

## Cellular Automaton Inspiration

This approach drew from cellular automaton principles, where local updates propagate globally, creating emergent behaviors suitable for pattern transformation tasks.

## Deterministic Nature

**PRK is a classical algorithm that produces reproducible results**. Unlike machine learning approaches that may rely on random chance, PRK's deterministic nature ensures consistent outcomes across multiple runs, making it highly reliable for pattern transformation tasks.

## Version 3 Developments

In the development of PRK v3, Chinese characters were adopted for visualization purposes instead of emojis due to GIF generation compatibility issues. These characters serve purely as symbolic representations and provide sufficient visual diversity within single character constraints to effectively display the various PRK states and transformations. Version 3 achieved correct annotation movement, though processing time may occasionally be extended for complex patterns.

## Strengths

- Excellent at identifying transformation priority zones
- Natural propagation mechanism for changes
- Robust local environment analysis
- Deterministic and reproducible results
- Classical algorithm approach ensuring stability

## Limitations

- Limited ability to abstract beyond training examples
- Difficulty in handling novel rule combinations
- Insufficient representation of meta-rules governing transformations
- Scaling constraints in version 2 implementation

## Legacy Impact

The PRK system provided valuable insights into cellular automaton-based approaches for pattern transformation, particularly in understanding local environment analysis and propagation mechanisms. These insights contributed to the development of more sophisticated representation systems.

---

**Research Status**: Archived - Superseded by Flat Space Representation approach

**Contact**: PIXMELT Research Team
