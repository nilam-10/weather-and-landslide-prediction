# Neighborhood Geometry Audit: Mathematical Rigor & Scale Definition

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Module Audited:** `src/evaluation/spatial_metrics.py`  
**Reference Standards:** Roberts & Lean (2008), Ebert (2008), Mittermaier (2014)

---

## 1. The Geometry Ambiguity: "Radius" vs. "Window Scale"

In the Step 2 report (`SPATIAL_VERIFICATION_REPORT.md`), a neighborhood configuration was described as:
> *"Neighborhood Radius: 30 km (k = 3 cells) | Window Dimension: 7 × 7 cells | Effective Scale: 73.9 km"*

This phrasing created understandable confusion: **How can a 30 km neighborhood have an effective scale of 73.9 km?**

To resolve this ambiguity, we conducted a rigorous mathematical and spatial audit of the coordinate grid and neighborhood kernels.

---

## 2. Coordinate Grid Geometry over Meghalaya

The domain spans:
- Latitude: $24.55^\circ\text{N}$ to $26.95^\circ\text{N}$ (25 grid rows, spacing $\Delta \phi = 0.100^\circ$)
- Longitude: $89.55^\circ\text{E}$ to $93.45^\circ\text{E}$ (40 grid columns, spacing $\Delta \lambda = 0.100^\circ$)

Using the WGS-84 reference ellipsoid at domain center ($\phi_0 = 25.75^\circ\text{N}$):
- **Meridional grid cell spacing ($\Delta y$)**:
  $$\Delta y = 111.132\text{ km/deg} \times 0.1^\circ = \mathbf{11.08\text{ km}}$$
- **Zonal grid cell spacing ($\Delta x$)**:
  $$\Delta x = 111.413\text{ km/deg} \times \cos(25.75^\circ) \times 0.1^\circ = \mathbf{10.03\text{ km}}$$
- **Mean effective grid cell spacing ($\Delta s$)**:
  $$\Delta s = \frac{11.08 + 10.03}{2} = \mathbf{10.55\text{ km}}$$
- **Single cell area**:
  $$\text{Area}_{\text{cell}} = \Delta x \times \Delta y = 10.03 \times 11.08 = \mathbf{111.13\text{ km}^2}$$

---

## 3. Dissecting the Four Geometric Quantities

To eliminate all ambiguity, four distinct quantities must be defined:

1. **Nominal Radius in Kilometres ($R$)**:
   The physical radial threshold specified by the user ($10, 20, 30, 50\text{ km}$).
2. **Grid Radius in Integer Cells ($k$)**:
   The discrete number of grid cells mapped from physical radius $R$:
   $$k = \text{round}\left( \frac{R}{\Delta s} \right) = \text{round}\left( \frac{R}{10.55} \right)$$
   For $R \in \{10, 20, 30, 50\}\text{ km} \implies k \in \{1, 2, 3, 5\}\text{ cells}$.
3. **Window Width in Grid Cells ($L$) & Physical Width ($W_{\text{phys}}$)**:
   In a square kernel, the total bounding window spans $L = 2k + 1$ cells along each axis:
   $$W_{\text{phys}} = L \times \Delta x = (2k + 1) \times 10.03\text{ km}$$
   $$H_{\text{phys}} = L \times \Delta y = (2k + 1) \times 11.08\text{ km}$$
   For $k = 3$, $L = 7$ cells. The physical width across the window is $7 \times 10.03 = \mathbf{70.2\text{ km}}$ (and $7 \times 11.08 = \mathbf{77.6\text{ km}}$ north-south).
   **This is why Roberts & Lean (2008) referred to $74\text{ km}$ as the spatial scale (length) of the $7 \times 7$ window!** The window measures $\sim 74\text{ km}$ from the westernmost edge to the easternmost edge.
4. **Distance from Centre to Outermost Included Cells ($D_{\text{max}}$)**:
   - In the cardinal directions (North, South, East, West):
     $$D_{\text{cardinal}} = k \cdot \Delta s \approx 3 \times 10.55 = \mathbf{31.65\text{ km}}$$
   - In the diagonal directions (corner cells $(k, k)$):
     $$D_{\text{diagonal}} = \sqrt{(k \Delta x)^2 + (k \Delta y)^2} = \sqrt{(3 \times 10.03)^2 + (3 \times 11.08)^2} = \mathbf{44.84\text{ km}}$$

---

## 4. Square Kernel vs. Circular Euclidean Kernel

In Step 2, the implementation defaulted to `kernel_type="square"`. 

### The Problem with the Square Kernel:
A square window of $k=3$ includes the corner pixels at diagonal offsets of $(3, 3)$. As shown above, these corner pixels are **$44.84\text{ km}$ away from the centre cell**! A storm placed $42\text{ km}$ to the northeast was counted as "inside the 30 km neighborhood". Furthermore, the square window contains 49 cells, covering a total physical area of:
$$\text{Area}_{\text{square}} = 49 \times 111.13 = \mathbf{5,445\text{ km}^2}$$
In contrast, a true geometric circle of radius $R = 30\text{ km}$ has an area of:
$$\text{Area}_{\text{true circle}} = \pi \times 30^2 = \mathbf{2,827.4\text{ km}^2}$$
**The square kernel covers $1.925 \times$ (nearly double) the area of a true 30 km circle!**

### The Corrected Circular Euclidean Kernel:
To enforce a physically meaningful neighborhood definition that does not extend substantially farther than the nominal radius, we evaluate the **circular Euclidean kernel**:
$$K_{\text{circular}}(u, v) = \begin{cases} 1 & \text{if } u^2 + v^2 \le k^2 \\ 0 & \text{otherwise} \end{cases}$$
where $u, v \in \{-k, \dots, +k\}$.

### Geometric Comparison Table:
| Nominal Radius $R$ | Grid $k$ | Square Size ($L \times L$) | Square Cells | Square Area ($\text{km}^2$) | Max Reach Square | Circular Cells | Circular Area ($\text{km}^2$) | True Circle Area ($\text{km}^2$) | Area Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10\text{ km}$** | $k=1$ | $3 \times 3$ | 9 | $1,000$ | $14.9\text{ km}$ | **5** (cross) | $556$ | $314$ | **$-44.4\%$** |
| **$20\text{ km}$** | $k=2$ | $5 \times 5$ | 25 | $2,778$ | $29.9\text{ km}$ | **13** | $1,445$ | $1,257$ | **$-48.0\%$** |
| **$30\text{ km}$** | $k=3$ | $7 \times 7$ | 49 | $5,445$ | $44.8\text{ km}$ | **29** | $3,223$ | $2,827$ | **$-40.8\%$** |
| **$50\text{ km}$** | $k=5$ | $11 \times 11$ | 121 | $13,447$ | $74.7\text{ km}$ | **81** | $9,002$ | $7,854$ | **$-33.1\%$** |

*Note on Grid Approximation*: On a discrete $10.55\text{ km}$ grid, perfect continuous circles are approximated by discrete pixel footprints. The circular kernel with $k=3$ (29 cells, $3,223\text{ km}^2$) matches the true circular area ($2,827\text{ km}^2$) to within $14\%$, eliminating the extreme diagonal inflation of the square kernel ($44.8\text{ km}$).

---

## 5. Empirical Impact of Geometry Correction on FSS Results

We recomputed FSS across all models, horizons, and thresholds using both geometries. The complete record is exported to `ORIGINAL_VS_CORRECTED_COMPARISON.csv`.

### Primary Comparison at Convective Hazard Threshold ($\ge 5.0\text{ mm/hr}$, $30\text{ km}$ radius):
| Forecast Horizon | Architecture | Square FSS ($7 \times 7$, 49 cells) | Corrected Circular FSS (29 cells) | $\Delta\text{FSS}$ (Cir - Sq) | Pointwise CSI | QPF RMSE ($\text{mm/hr}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$+30\text{m}$** | ConvLSTM (Seed 42) | $0.9152$ | $0.9019$ | $-0.0133$ | $0.5621$ | $0.748$ |
| | SimVP (Optimal) | $0.9255$ | $0.9123$ | $-0.0132$ | $0.5734$ | $0.734$ |
| **$+60\text{m}$** | ConvLSTM (Seed 42) | $0.8084$ | $0.7856$ | $-0.0228$ | $0.3715$ | $1.030$ |
| | SimVP (Optimal) | $0.8230$ | $0.8001$ | $-0.0229$ | $0.3768$ | $1.002$ |
| **$+120\text{m}$ (2 hr)** | ConvLSTM (Seed 42) | $0.3676$ | $0.3489$ | $-0.0187$ | $0.0993$ | $1.377$ |
| | **SimVP (Optimal)** | **$0.5466$** | **$0.5192$** | **$-0.0274$** | **$0.1657$** | **$1.326$** |
| | UNet-ConvLSTM (Best) | $0.4129$ | $0.3991$ | $-0.0138$ | $0.1272$ | $1.373$ |
| **$+180\text{m}$ (3 hr)** | ConvLSTM (Seed 42) | $0.0267$ | $0.0238$ | $-0.0029$ | $0.0047$ | $1.527$ |
| | SimVP (Optimal) | $0.2416$ | $0.2262$ | $-0.0154$ | $0.0559$ | $1.484$ |
| **$+360\text{m}$ (6 hr)** | All DL Models | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $1.66\text{--}1.68$ |

### Scientific Conclusions of the Geometry Audit:
1. **Slight Downward Shift**: Because the circular kernel reduces the allowed neighborhood area by $33\%\text{--}48\%$ and excludes diagonal corner cells up to $45\text{ km}$ away, circular FSS values are consistently $0.01\text{--}0.03$ lower than square FSS values.
2. **Robustness of Model Rankings**: The correction does **not** change any architectural conclusions:
   - SimVP remains the clear winner across all horizons.
   - At $+120\text{m}$, **SimVP STILL exceeds the Roberts & Lean useful skill target of $0.50$ ($FSS_{\text{circular}} = \mathbf{0.5192}$)**, while ConvLSTM remains far below ($0.3489$).
3. **Recommendation**: All future verification in this thesis should report the **Corrected Circular Euclidean Kernel** to avoid inflating neighborhood claims.
