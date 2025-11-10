# Distance-Based Attention System for ARC Prize

**Historical Approach - PIXMELT Research**

*Note: This document was originally written in French and translated to English for broader accessibility. All content has been reviewed and validated by the research team.*

---

## Overview

The Distance-Based Attention System was an experimental approach for solving ARC Prize challenges through novel attention mechanisms based on spatial distances between grid positions. The system operated through multiple agents that progressively transformed inputs by adding transformation layers.

## Core Architecture

```python
class GradLinExpert(nn.Module):
    VERSION = '1.0.0'
    def __init__(
        self,
        lsize:int = 9,
        num_classes:int = 11,
    ):
        super().__init__()
        self.params = [ lsize, num_classes ]
        # Props
        self.register_buffer('lsize', torch.tensor(lsize, dtype=torch.int))
        self.register_buffer('num_classes', torch.tensor(num_classes, dtype=torch.long))
        # Weights
        self.encoder = nn.Parameter(torch.ones(lsize, num_classes))
        self.attention = nn.Parameter(torch.rand(lsize * (lsize - 1) // 2) * (num_classes-1))
        self.decoder = nn.Parameter(torch.ones(lsize * (lsize - 1) // 2, num_classes))
        # Caches
        self.attention_pos = self.get_pairwise_positions(lsize)

    def forward(self, inputs):
        e_min = 1e-8
        self.attention_pos = self.attention_pos.to(inputs.device)

        # Encoder
        encoded_values = inputs * self.encoder

        # Attention
        def compute_distance(a, b, c):
            return torch.sqrt((a-b)**2 + (a-c)**2 + e_min)
        attention = compute_distance(
            self.attention[:, None], # Target
            encoded_values[:, self.attention_pos[:, 0]], # Left
            encoded_values[:, self.attention_pos[:, 1]], # Right
        )
        at_max = torch.amax(attention, dim=(1, 2), keepdim=True)
        attention = attention / (at_max + e_min)
        attention.neg_().add_(1)

        # Decoder
        attention.mul_(self.decoder)

        # Projection
        projection = torch.zeros(inputs.shape[0], self.lsize, self.num_classes, device=inputs.device)
        projection.scatter_add_(1, self.attention_pos[None, :, 0, None].expand(attention.shape), attention)
        projection.scatter_add_(1, self.attention_pos[None, :, 1, None].expand(attention.shape), attention)
        pj_max = torch.amax(projection, dim=(1, 2), keepdim=True)
        projection = projection / (pj_max + e_min)
        return projection
```

## Key Mechanisms

### Distance-Based Attention Formula

The attention mechanism computes distances between coupled values using:
```
distance = sqrt((target - left)² + (target - right)² + ε)
```

Where positions are paired systematically (e.g., for 2×2 grid: (0,0)↔(1,0), (0,0)↔(0,1), (0,0)↔(1,1), then (1,0)↔(0,1), etc.)

### Layer Calculation Formula

The number of transformation layers was determined by:
```
layers = 0.4 × distance_max
```

Where distance_max represents the maximum number of pixel changes between input and target grids.

## Demonstration Results

### Attention Heat Map

<img src="demos/distance_attention_examples/attention.gif" width="250" alt="Attention Heat Map">

*Attention heat map visualization showing the distance-based attention mechanism in action. Brighter areas indicate higher attention values between coupled grid positions.*

### Version 2 Results - Scaling Improvements

**Puzzle ID: 00d62c1b**  
<img src="demos/distance_attention_examples/v2-00d62c1b.gif" width="250" alt="Distance Attention v2 - 00d62c1b">

**Puzzle ID: 09629e4f**  
<img src="demos/distance_attention_examples/v2-09629e4f.gif" width="250" alt="Distance Attention v2 - 09629e4f">

*Progressive transformations showing improved scaling capabilities across different grid sizes.*

### Version 3 Results - Enhanced Precision and Layered System

**Puzzle ID: 0e671a1a**

<img src="demos/distance_attention_examples/v3-0e671a1a-ex0.gif" width="250" alt="v3 Example 0">
<img src="demos/distance_attention_examples/v3-0e671a1a-ex1.gif" width="250" alt="v3 Example 1">
<img src="demos/distance_attention_examples/v3-0e671a1a-ex2.gif" width="250" alt="v3 Example 2">
<img src="demos/distance_attention_examples/v3-0e671a1a-ex3.gif" width="250" alt="v3 Example 3">
<img src="demos/distance_attention_examples/v3-0e671a1a-t0.gif" width="250" alt="v3 Test Case">

*Step-by-step evolution showing improved trait precision and multi-layer processing system for enhanced pattern recognition and rule application.*

## Strengths

- Successfully captured spatial relationships between grid elements
- Effective at reproducing specific transformation rules
- Computationally efficient for local pattern recognition

## Limitations

While this approach successfully reproduced certain complex ARC rules, it failed to generalize beyond training examples, suggesting that distance-based attention alone was insufficient for abstract reasoning.

## Legacy Impact

The insights gained from this distance-based attention system contributed to the development of more sophisticated approaches, particularly in understanding the importance of spatial relationships and the need for better generalization mechanisms in ARC solving systems.

---

**Research Status**: Archived - Superseded by Flat Space Representation approach

**Contact**: PIXMELT Research Team
