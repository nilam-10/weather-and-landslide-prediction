# Mathematical Formulations and Derivations

**Project Title**: Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Institutional Context**: RBU Honours Major Project (2026–27)

---

## 1. Convective Weather Nowcasting Formulations

### 1.1 Spatiotemporal ConvLSTM Recurrent Cell
Let $\mathcal{X}_t \in \mathbb{R}^{C \times H \times W}$ represent the multi-channel satellite observation tensor at time step $t$, where channel $c=0$ is the normalized MERGIR infrared brightness temperature $T_b$ and $c=1$ is the normalized IMERG precipitation rate $P$.

The ConvLSTM transitions from hidden state $\mathcal{H}_{t-1} \in \mathbb{R}^{D \times H \times W}$ and cell state $\mathcal{C}_{t-1} \in \mathbb{R}^{D \times H \times W}$ to $(\mathcal{H}_t, \mathcal{C}_t)$ via:
$$\begin{aligned}
\mathcal{I}_t &= \sigma\left(\mathcal{W}_{xi} \ast \mathcal{X}_t + \mathcal{W}_{hi} \ast \mathcal{H}_{t-1} + \mathcal{W}_{ci} \odot \mathcal{C}_{t-1} + b_i\right) \\
\mathcal{F}_t &= \sigma\left(\mathcal{W}_{xf} \ast \mathcal{X}_t + \mathcal{W}_{hf} \ast \mathcal{H}_{t-1} + \mathcal{W}_{cf} \odot \mathcal{C}_{t-1} + b_f\right) \\
\tilde{\mathcal{C}}_t &= \tanh\left(\mathcal{W}_{xc} \ast \mathcal{X}_t + \mathcal{W}_{hc} \ast \mathcal{H}_{t-1} + b_c\right) \\
\mathcal{C}_t &= \mathcal{F}_t \odot \mathcal{C}_{t-1} + \mathcal{I}_t \odot \tilde{\mathcal{C}}_t \\
\mathcal{O}_t &= \sigma\left(\mathcal{W}_{xo} \ast \mathcal{X}_t + \mathcal{W}_{ho} \ast \mathcal{H}_{t-1} + \mathcal{W}_{co} \odot \mathcal{C}_t + b_o\right) \\
\mathcal{H}_t &= \mathcal{O}_t \odot \tanh\left(\mathcal{C}_t\right)
\end{aligned}$$
where:
- $\ast$ denotes the 2D spatial convolution operator with kernel $k \times k$ (here $3 \times 3$),
- $\odot$ denotes the Hadamard element-wise product,
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the element-wise logistic sigmoid function,
- $\mathcal{I}_t, \mathcal{F}_t, \mathcal{O}_t$ denote the input, forget, and output gate activation tensors respectively.

### 1.2 Multi-Horizon Output Projection Heads
From the final recurrent state $\mathcal{H}_{T_{\text{in}}}$, multi-lead prediction heads project features to convective rainfall probabilities at horizons $\tau \in \{+30\text{m}, +60\text{m}\}$:
$$\hat{\mathcal{Y}}_\tau = \sigma\left(\mathcal{W}_{\text{proj}, \tau}^{(2)} \ast \operatorname{ReLU}\left(\mathcal{W}_{\text{proj}, \tau}^{(1)} \ast \mathcal{H}_{T_{\text{in}}} + b_{\tau}^{(1)}\right) + b_{\tau}^{(2)}\right)$$
where $\hat{\mathcal{Y}}_\tau(i, j) \in [0, 1]$ represents the estimated probability $P\left(P_{t+\tau}(i, j) \ge 5.0\text{ mm/hr} \mid \mathcal{X}_{1:t}\right)$.

### 1.3 Compound Weighted Nowcasting Loss
Because convective rainfall events are spatially sparse (high class imbalance: typically $< 15\%$ positive pixels), standard binary cross-entropy induces mode collapse toward predicting all-zero rasters. We optimize a **Weighted Binary Cross-Entropy Loss** with positive class penalty $w_{\text{pos}} = 2.5$:
$$\mathcal{L}_{\tau}(y, \hat{y}) = -\frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W \left[ w_{\text{pos}} \cdot y_{ij} \log(\hat{y}_{ij}) + (1 - y_{ij}) \log(1 - \hat{y}_{ij}) \right]$$
The multi-task objective function combines horizons with decaying weights $\alpha_1 = 0.6$ (+30m) and $\alpha_2 = 0.4$ (+60m):
$$\mathcal{L}_{\text{total}} = \alpha_1 \mathcal{L}_{+30\text{m}} + \alpha_2 \mathcal{L}_{+60\text{m}} + \lambda_{\text{reg}} \|\Theta\|_2^2$$

### 1.4 Verification Metrics for Categorical Precipitation Nowcasting
Given binary threshold $T = 5.0\text{ mm/hr}$ and probability decision threshold $\theta = 0.5$:
- $\text{Hits (TP)} = \sum \mathbb{I}(\hat{y} \ge \theta \land y = 1)$
- $\text{False Alarms (FP)} = \sum \mathbb{I}(\hat{y} \ge \theta \land y = 0)$
- $\text{Misses (FN)} = \sum \mathbb{I}(\hat{y} < \theta \land y = 1)$
- $\text{Correct Negatives (TN)} = \sum \mathbb{I}(\hat{y} < \theta \land y = 0)$

1. **Critical Success Index (CSI / Threat Score)**:
   $$\text{CSI} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} \in [0, 1]$$
2. **Probability of Detection (POD / Hit Rate)**:
   $$\text{POD} = \frac{\text{TP}}{\text{TP} + \text{FN}} \in [0, 1]$$
3. **False Alarm Ratio (FAR)**:
   $$\text{FAR} = \frac{\text{FP}}{\text{TP} + \text{FP}} \in [0, 1]$$
4. **Heidke Skill Score (HSS)** (skill relative to random chance):
   $$\text{HSS} = \frac{2(\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN})}{(\text{TP} + \text{FN})(\text{FN} + \text{TN}) + (\text{TP} + \text{FP})(\text{FP} + \text{TN})}$$
5. **Brier Score (BS)** (probabilistic calibration accuracy):
   $$\text{BS} = \frac{1}{N} \sum_{k=1}^N (\hat{p}_k - y_k)^2 \in [0, 1]$$
   where $0$ indicates perfect deterministic probability assignment.

---

## 2. Optical Flow Kinematic Extrapolation Mathematics

### 2.1 Optical Flow Constraint Equation
Assuming the precipitation field $I(x, y, t)$ satisfies the brightness constancy hypothesis under advection:
$$\frac{d I}{d t} = \frac{\partial I}{\partial x}\frac{d x}{d t} + \frac{\partial I}{\partial y}\frac{d y}{d t} + \frac{\partial I}{\partial t} = 0$$
Let $u = \frac{dx}{dt}$ and $v = \frac{dy}{dt}$ represent velocity vector components. Then:
$$I_x u + I_y v + I_t = 0$$

### 2.2 Farneback Polynomial Expansion
The neighborhood of each pixel in two consecutive frames is approximated by quadratic polynomials:
$$f_1(\mathbf{x}) = \mathbf{x}^T \mathbf{A}_1 \mathbf{x} + \mathbf{b}_1^T \mathbf{x} + c_1$$
$$f_2(\mathbf{x}) = (\mathbf{x} - \mathbf{d})^T \mathbf{A}_1 (\mathbf{x} - \mathbf{d}) + \mathbf{b}_1^T (\mathbf{x} - \mathbf{d}) + c_1 = \mathbf{x}^T \mathbf{A}_2 \mathbf{x} + \mathbf{b}_2^T \mathbf{x} + c_2$$
Equating coefficients yields the local displacement $\mathbf{d} = (u, v)^T$:
$$2 \mathbf{A} \mathbf{d} = -(\mathbf{b}_2 - \mathbf{b}_1)$$
where $\mathbf{A} = \frac{\mathbf{A}_1 + \mathbf{A}_2}{2}$.

### 2.3 Semi-Lagrangian Backward Advection
For future lead step $\tau \in \{1, 2, 4, 6\}$ (corresponding to $+30\text{m}, +60\text{m}, +120\text{m}, +180\text{m}$):
$$\hat{I}_{t+\tau}(x, y) = I_t(x - u(x, y) \cdot \tau, \; y - v(x, y) \cdot \tau)$$
Interpolation at non-grid coordinates is performed using bilinear interpolation with zero boundary conditions outside the domain buffer.

---

## 3. Topographic Geomorphometry Mathematical Derivations

### 3.1 Local Surface Fitting on DEM
Given a $3 \times 3$ elevation window centered at cell $z_5$ with grid spacing $g$:
$$\begin{bmatrix} z_1 & z_2 & z_3 \\ z_4 & z_5 & z_6 \\ z_7 & z_8 & z_9 \end{bmatrix}$$
The continuous surface $z(x, y)$ is parameterized following Zevenbergen & Thorne (1987):
$$z(x, y) = \frac{a x^2}{2} + \frac{b y^2}{2} + c x y + d x + e y + f$$
Partial derivatives are computed via central finite differences:
$$p = \frac{\partial z}{\partial x} = \frac{z_6 - z_4}{2g}, \quad q = \frac{\partial z}{\partial y} = \frac{z_2 - z_8}{2g}$$
$$r = \frac{\partial^2 z}{\partial x^2} = \frac{z_4 - 2z_5 + z_6}{g^2}, \quad t = \frac{\partial^2 z}{\partial y^2} = \frac{z_2 - 2z_5 + z_8}{g^2}$$
$$s = \frac{\partial^2 z}{\partial x \partial y} = \frac{-z_1 + z_3 + z_7 - z_9}{4g^2}$$

### 3.2 Terrain Morphometric Attributes
1. **Slope Angle ($\beta$)**:
   $$\beta = \arctan\left(\sqrt{p^2 + q^2}\right) \cdot \frac{180}{\pi} \quad [\text{degrees}]$$
2. **Aspect Angle ($\theta$)**:
   $$\theta = 180 - \arctan\left(\frac{q}{p}\right) + 90 \cdot \frac{p}{|p|} \quad [\text{degrees clockwise from North}]$$
3. **Continuous Cyclical Aspect Components**:
   $$\text{Northness} = \cos\left(\frac{\theta \pi}{180}\right) \in [-1, 1], \quad \text{Eastness} = \sin\left(\frac{\theta \pi}{180}\right) \in [-1, 1]$$
4. **Profile Curvature ($K_{\text{prof}}$)** (rate of change of slope along flowline):
   $$K_{\text{prof}} = -200 \cdot \frac{p^2 r + 2pq s + q^2 t}{(p^2 + q^2)(1 + p^2 + q^2)^{3/2}} \quad [100/\text{m}]$$
5. **Planform Curvature ($K_{\text{plan}}$)** (curvature along contour, governs flow convergence):
   $$K_{\text{plan}} = 200 \cdot \frac{q^2 r - 2pq s + p^2 t}{(p^2 + q^2)^{3/2}} \quad [100/\text{m}]$$

---

## 4. Landslide Susceptibility Mathematical Formulations

### 4.1 Gradient Boosted Decision Trees (XGBoost)
The ensemble prediction for sample $\mathbf{x}_i \in \mathbb{R}^D$ is the sum of $K$ regression trees:
$$\hat{y}_i = \phi(\mathbf{x}_i) = \sum_{k=1}^K f_k(\mathbf{x}_i), \quad f_k \in \mathcal{F}$$
At step $m$, the objective minimized is:
$$\mathcal{L}^{(m)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(m-1)} + f_m(\mathbf{x}_i)\right) + \Omega(f_m)$$
where tree complexity penalty is $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$.

Taking the second-order Taylor expansion:
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ l(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
where first and second order gradients are:
$$g_i = \frac{\partial l(y_i, \hat{y}_i^{(m-1)})}{\partial \hat{y}_i^{(m-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(m-1)})}{\partial (\hat{y}_i^{(m-1)})^2}$$
For logistic loss with positive weight $w_{\text{pos}}$:
$$g_i = \hat{p}_i - y_i, \quad h_i = \hat{p}_i (1 - \hat{p}_i)$$
The optimal leaf weight for leaf $j$ containing sample index set $I_j$ is analytically:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
and the optimal split gain is:
$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

## 5. Dynamic Rainfall Hazard Formulation (NASA LHASA 2.0)

### 5.1 Antecedent Precipitation Integration
Cumulative antecedent precipitation over backward time window $\Delta T$ from reference time $t_0$:
$$P_{\Delta T}(x, y, t_0) = \int_{t_0 - \Delta T}^{t_0} P(x, y, \tau) \, d\tau \approx \sum_{k=0}^{K-1} P(x, y, t_0 - k \cdot \delta t) \cdot \delta t$$
where $\delta t = 0.5\text{ hr}$, evaluating $\Delta T \in \{24\text{h}, 72\text{h}\}$.

### 5.2 Dynamic Hazard Level Assignment
Let $S(x, y) \in [0, 1]$ be the Static Landslide Susceptibility Index, and $A(x, y) = 0.6 P_{24\text{h}} + 0.4 P_{72\text{h}}$ be the weighted antecedent rainfall trigger. The dynamic hazard level $H(x, y) \in \{0, 1, 2, 3\}$ is defined by the piecewise step function:
$$H(x, y) = \begin{cases}
3 \quad (\text{Very High}), & \text{if } (S \ge 0.6 \land A \ge 100\text{ mm}) \lor (S \ge 0.8 \land A \ge 50\text{ mm}) \\
2 \quad (\text{High}),      & \text{if } (S \ge 0.6 \land 50 \le A < 100\text{ mm}) \lor (S \ge 0.3 \land A \ge 100\text{ mm}) \\
1 \quad (\text{Moderate}),  & \text{if } (S \ge 0.8 \land A < 50\text{ mm}) \lor (0.3 \le S < 0.6 \land 50 \le A < 100\text{ mm}) \lor (S < 0.3 \land A \ge 100\text{ mm}) \\
0 \quad (\text{Low}),       & \text{otherwise}
\end{cases}$$
