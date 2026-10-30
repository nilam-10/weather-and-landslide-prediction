# Teacher Demonstration & Viva Examination Guide

**Project Title:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Candidate Name:** B.Tech Honours Major Project Candidate (2026–27)  
**Institution:** Royal Global University  
**Demonstration URL:** `http://127.0.0.1:8000/`  
**Guide File Path:** `outputs/reports/local_run_and_code_guide/TEACHER_DEMONSTRATION_GUIDE.md`  

---

## 1. Pre-Demonstration Checklist

Before inviting your teacher or viva examiner to view your screen:
1. Confirm the local server is running in your terminal:
   ```powershell
   # Confirm output shows: Uvicorn running on http://127.0.0.1:8000
   ```
2. Open Google Chrome or Microsoft Edge and navigate to:
   $$\mathbf{http://127.0.0.1:8000/}$$
3. Press `F11` (optional) to maximize the browser for a clean, professional full-screen presentation.
4. Have the visual figures ready in your thesis folder or image viewer:
   - `C:\Users\nilam\.gemini\antigravity\brain\e65f9d73-ffd7-43d4-8268-df50ca1388f3\fig_architecture_comparison_csi.png`
   - `C:\Users\nilam\.gemini\antigravity\brain\e65f9d73-ffd7-43d4-8268-df50ca1388f3\fig_multi_model_prediction_maps.png`

---

## 2. Step-by-Step Live Demonstration Script

### Act 1: The Overview & Scientific Context (2 Minutes)
- **What to Say:**
  > *"Good morning, Professor. Today I am demonstrating our B.Tech Honours Major Project: a Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction tailored specifically for Meghalaya, Northeast India.*  
  > *Meghalaya experiences some of the highest monsoon rainfall in the world, leading to frequent flash floods and slope failures. Our system integrates multi-satellite observations—specifically NASA GPM IMERG precipitation and MERGIR thermal infrared—with high-resolution SRTM digital elevation geomorphometry to deliver real-time, multi-horizon weather nowcasting and dynamic hazard visualization."*
- **What to Show on Screen:**
  - Point to the central Leaflet Web GIS map displaying the Meghalaya boundary outline, the Shillong Plateau, and the Cherrapunji / Sohra escarpments.
  - Point to the System Health indicator in the top navbar displaying: **Status: Online | RTX 4050 GPU (CUDA) | 336 Monsoon Timesteps Ready**.

---

### Act 2: Demonstrating Weather Nowcasting (3 Minutes)
1. **Selecting a Convective Storm Timestamp:**
   - Move the timeline slider to **Step 316** (July 7, 2025, 14:00 UTC).
   - Point out the intense convective storm signature visible over the southern escarpment.
2. **Demonstrating Multi-Horizon Forecasting:**
   - In the **Lead Time** dropdown, start at **+30m**.
   - Show how the model predicts the immediate forward propagation of the storm cell.
   - Switch the dropdown to **+60m**, then **+120m**, and **+180m**.
   - **What to Explain:**
     > *"As we advance the forecast horizon from 30 minutes to 3 hours, you can see the neural model advect the rain field eastward while progressively dissipating intensity. This matches physical mesoscale convective dynamics, where storm cores weaken over time without continuous latent heat replenishment."*
3. **Comparing with Baseline Models:**
   - Switch the **Model** dropdown from **ConvLSTM** to **Optical Flow (Lucas-Kanade)**.
   - Show how Optical Flow merely pushes pixels in a straight line without modeling cell growth or decay.
   - Switch to **Persistence (Lag Hold)**.
   - Show how Persistence leaves the rain frozen in place, completely missing storm movement.
   - Switch back to **ConvLSTM** to highlight the superior shape deformation learned by the neural network.

---

### Act 3: Multimodal Cell Inspector (2 Minutes)
1. **Interactive Map Click:**
   - Click directly on the map near **Cherrapunji / Sohra** ($25.27^\circ\text{N}, 91.73^\circ\text{E}$).
   - The right-hand **Cell Inspector** sidebar immediately populates with dual-stack diagnostics.
2. **Explain the Inspected Data:**
   - **Meteorological State:** Shows current rainfall rate ($18.4\text{ mm/hr}$), thermal IR cloud-top temperature ($218\text{ K}$, indicating deep convective cloud tops), and the 5-point predicted rainfall curve.
   - **Terrain & Susceptibility State:** Shows SRTM slope ($28.4^\circ$), Topographic Wetness Index, and static Landslide Susceptibility Index ($0.74$, Tier: High Susceptibility).
   - **Dynamic Hazard:** Shows LHASA 2.0 Dynamic Hazard level (**Moderate Hazard / Tier 2**).

---

### Act 4: Landslide Risk Layers & The Co-Location Feature (2 Minutes)
1. **Toggle Static Susceptibility:**
   - Open the **Layers** panel and check **90m Landslide Susceptibility**.
   - Explain how machine learning (XGBoost & Random Forest) mapped steep escarpment slopes into high-susceptibility zones (orange/red).
2. **Toggle Historical Landslides:**
   - Check **Historical Landslide Events**.
   - 38 verified Geological Survey of India (GSI) event markers appear on the map.
   - Click on an event marker to show verified historical date, trigger rainfall, and coordinates.
3. **Toggle Spatial Co-Location:**
   - Check **Spatial Co-Location**.
   - Explain the yellow highlighted intersecting cells and point out the **Anti-Fallacy Notice**:
     > *"Notice this prominent disclaimer in our platform: 'Independent Co-Location Only — Not a Joint Probability'. We deliberately do not multiply weather probability by landslide susceptibility, because mathematically, multiplying two uncalibrated marginal probabilities violates Bayes' theorem and creates a false sense of precision. We present them as side-by-side decision support layers."*

---

### Act 5: Explainable AI (XAI) Feature (1 Minute)
1. In the sidebar, click **"Compute Model Attribution"**.
2. Within 1.5 seconds, the PyTorch Integrated Gradients algorithm executes on the RTX 4050 GPU.
3. Show the attribution heatmap overlay.
4. **What to Explain:**
   > *"This XAI map reveals which historical satellite frames and geographic quadrants most heavily influenced the neural network's rainfall prediction. The strong positive gradients along the southern moisture-inflow corridor prove the network has learned physically meaningful convective inflow rather than overfitting to spurious background noise."*

---

## 3. Explaining the Research Figures to Your Teacher

When presenting your research dissertation chapters, pull up these two key figures:

### Figure 1: Architecture Comparison (CSI Across Lead Times)
- **Artifact:** `fig_architecture_comparison_csi.png`
- **What the Graph Shows:**
  The X-axis represents forecast lead time (+30m to +360m); the Y-axis shows the Critical Success Index (CSI) for convective rain ($\ge 5\text{ mm/hr}$).
- **How to Explain It:**
  > *"This graph compares our canonical ConvLSTM against the experimental SimVP, U-Net–ConvLSTM, and classical Optical Flow. At +30 minutes, all neural models outperform Optical Flow ($0.31$ vs $0.24$). At intermediate horizons (+60m and +120m), the non-recurrent SimVP architecture demonstrates greater skill retention because its convolutional Translator avoids recurrent state blurring. However, beyond 3 hours, all models experience steep convective decorrelation, reaching near-zero skill at +360 minutes."*

### Figure 2: Multi-Model Spatial Prediction Maps
- **Artifact:** `fig_multi_model_prediction_maps.png`
- **What the Graph Shows:**
  A 4-column visual grid comparing Ground Truth IMERG observations against predictions from ConvLSTM, SimVP, and Optical Flow across the Meghalaya terrain.
- **How to Explain It:**
  > *"Here we visually verify the spatial forecast quality. Notice that Optical Flow simply shifts the initial rain blob without accounting for topography. ConvLSTM preserves spatial boundaries well but tends to smooth out intense peak cores. SimVP retains sharper rain gradients, but when trained with standard MSE, it suffered from spatial displacement. This finding motivated our Step 4 and Step 5 research into multiscale Soft-FSS loss functions."*

---

## 4. Top 10 Likely Viva Questions & Model Answers

### Q1: Why did you choose ConvLSTM instead of a standard 3D CNN or Vision Transformer?
**Answer:**  
*"ConvLSTM combines the spatial feature extraction of 2D convolutions with the temporal state-tracking of recurrent LSTM cells. While 3D CNNs have rigid temporal receptive fields and Vision Transformers require tens of thousands of training samples that can easily overfit on regional monsoon datasets, ConvLSTM provides an ideal inductive bias for fluid-like advection with a compact footprint of only 161,034 parameters, making it trainable and deployable on edge/laptop GPUs."*

---

### Q2: What are the input channels to your weather model, and why did you choose them?
**Answer:**  
*"We use three complementary channels:  
1. **NASA MERGIR 4km Thermal Infrared (TIR-1):** Measures cloud-top brightness temperature. Rapid cooling indicates vigorous vertical convective updrafts.  
2. **NASA GPM IMERG Half-Hourly Precipitation Rate:** Ground-calibrated satellite rainfall providing the current surface precipitation state.  
3. **Satellite Observation Validity Mask:** A binary quality flag that allows the network to ignore missing satellite scans or border artifacts without corrupting convolutional activations."*

---

### Q3: Why is standard Mean Squared Error (MSE) problematic for rainfall forecasting?
**Answer:**  
*"Precipitation is highly sparse: over $90\%$ of grid cells have zero rain, while extreme convective cells exceed $25\text{ mm/hr}$. Minimizing MSE causes the network to predict a blurry, low-intensity spatial average to hedge against location errors (the 'double-penalty' problem). This eliminates the heavy rain peaks that are most hazardous. To solve this, our research incorporated an intensity-weighted L1 loss and a differentiable multiscale Fractions Skill Score (Soft-FSS)."*

---

### Q4: Why does forecasting skill drop drastically at 6 hours (+360 minutes)?
**Answer:**  
*"Convective thunderstorms over the Meghalaya plateau have physical lifecycles of 30 to 90 minutes. Satellite-based nowcasting extrapolates existing cloud and precipitation dynamics. At 6 hours, new convective cells initiate and dissipate due to synoptic wind fields and thermodynamics that cannot be observed from local satellite extrapolations alone. Reaching skill at 6 hours requires coupling with numerical weather prediction (NWP) models like WRF or ECMWF."*

---

### Q5: How did you validate your landslide susceptibility model, and what was its performance?
**Answer:**  
*"We trained XGBoost, Random Forest, and Logistic Regression on 5 geomorphometric features derived from 90m SRTM elevation data. Crucially, we avoided standard random k-fold cross-validation, which overestimates accuracy due to spatial autocorrelation. Instead, we implemented **spatial block cross-validation**, dividing Meghalaya into geographically disjoint blocks. Our models achieved an honest spatial ROC-AUC of $0.5913 \pm 0.04$, realistically reflecting the difficulty of regional landslide prediction with sparse historical inventories ($N=38$ events)."*

---

### Q6: What is Platt Scaling, and why is it necessary in your pipeline?
**Answer:**  
*"Because we trained the neural network with a weighted cross-entropy loss ($\text{pos\_weight} \approx 20$) to force attention onto rare convective events, the raw output logits are systematically shifted toward overprediction. Platt Scaling fits an empirical post-hoc logistic regression (slope $a$ and intercept $b$) on an independent validation set. It maps raw logits back to true, well-calibrated probabilities, significantly improving the Brier reliability score without altering model weights."*

---

### Q7: Why do you refuse to multiply rainfall probability and landslide susceptibility into a single score?
**Answer:**  
*"In our platform, we enforce an explicit **Anti-Fallacy Policy**. In literature, some naive systems compute $\text{Risk} = P(\text{Rain}) \times P(\text{Slide})$. This is mathematically invalid because the two models operate on vastly different spatial resolutions ($10\text{ km}$ vs $90\text{ m}$), have different calibration baselines, and are not conditionally independent. Multiplying them creates a false sense of mathematical precision. Instead, we present them as an objective co-location overlay and a 2D decision matrix inspired by NASA LHASA 2.0."*

---

### Q8: How did you implement Explainable AI (XAI) in your weather model?
**Answer:**  
*"We implemented **Integrated Gradients** (Sundararajan et al., 2017). We establish a blank baseline tensor (zero rain, neutral cloud temperature) and compute path integrals of the output prediction gradients along 10 interpolation steps using PyTorch autograd. This identifies exactly which geographic grid cells and past observation timesteps drove the nowcast, providing transparent scientific attribution."*

---

### Q9: What is the difference between canonical ConvLSTM and the experimental SimVP model you researched?
**Answer:**  
*"ConvLSTM is recurrent: it sequentially passes hidden states through convolutional gates. SimVP (Simpler Video Prediction) is completely non-recurrent: it uses a spatial CNN encoder, a temporal Translator built from stacked 2D convolutions, and a spatial decoder. In our Step 5 three-seed replication, SimVP with our combined loss improved +120m CSI by $+20\%$ over baseline. However, to maintain rock-solid system stability, our live production server runs the canonical ConvLSTM checkpoint."*

---

### Q10: If you had another 6 months on this project, what would be your next priority?
**Answer:**  
*"My primary priority would be **multimodal NWP fusion**: coupling satellite nowcasts with coarse-resolution boundary conditions from NCMRWF or IMD WRF numerical weather models to extend convective skill beyond the 3-hour limit. Second, I would integrate real-time soil moisture retrievals (such as NASA SMAP or CYGNSS) to replace empirical antecedent rainfall proxies with true root-zone physical moisture modeling."*
