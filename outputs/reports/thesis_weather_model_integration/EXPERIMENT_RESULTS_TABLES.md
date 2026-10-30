# Authoritative Experiment Results Tables: Weather Nowcasting Research

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Region:** Meghalaya Plateau, Northeast India ($24.55^\circ\text{N} - 26.95^\circ\text{N}, 89.55^\circ\text{E} - 93.45^\circ\text{E}$)  
**Grid:** $25 \times 40$ cells ($0.10^\circ \approx 10.55\text{ km}$ spacing)  
**Primary Datasets:** 38-Day Training ($N=1,764$ seqs), July 5, 2025 Validation ($N=33$ seqs), July 12–14, 2023 Confirmatory Benchmark ($N=129$ seqs)  
**Status:** **AUTHORITATIVE RECONCILED TABLES FOR B.TECH THESIS**

---

## Table 1: Validation Results Across Seeds (July 5, 2025, $N=33$ Sequences)

Evaluated at the convective precipitation threshold $\ge 5.0\text{ mm/hr}$. All metrics report Mean $\pm$ Standard Deviation across $N=3$ random initializations (Seeds 42, 123, 999).

| Horizon | Metric | Units | Baseline (Exp A) | Combined (Exp D) | Absolute Gain ($\Delta$) | Relative Change |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+30m** | Pointwise CSI | dimensionless | $0.3491 \pm 0.0263$ | $\mathbf{0.3819 \pm 0.0160}$ | $+0.0328$ | $+9.4\%$ |
| | Circular FSS ($30\text{ km}$) | dimensionless | $0.7583 \pm 0.0464$ | $\mathbf{0.7858 \pm 0.0121}$ | $+0.0275$ | $+3.6\%$ |
| | Probability of Detection (POD) | dimensionless | $0.4454 \pm 0.0521$ | $\mathbf{0.5658 \pm 0.0702}$ | $+0.1204$ | $+27.0\%$ |
| | False Alarm Ratio (FAR) | dimensionless | $\mathbf{0.3766 \pm 0.0377}$ | $0.4545 \pm 0.0291$ | $+0.0779$ | $+20.7\%$ |
| | QPF RMSE | $\text{mm/hr}$ | $\mathbf{1.1150 \pm 0.0119}$ | $1.1531 \pm 0.0485$ | $+0.0381$ | $+3.4\%$ |
| | QPF MAE | $\text{mm/hr}$ | $\mathbf{0.3979 \pm 0.0116}$ | $0.4348 \pm 0.0227$ | $+0.0369$ | $+9.3\%$ |
| | Pearson Correlation ($r$) | dimensionless | $\mathbf{0.7714 \pm 0.0038}$ | $0.7678 \pm 0.0036$ | $-0.0036$ | $-0.5\%$ |
| **+60m** | Pointwise CSI | dimensionless | $0.2094 \pm 0.0100$ | $\mathbf{0.2662 \pm 0.0227}$ | $+0.0568$ | $+27.1\%$ |
| | Circular FSS ($30\text{ km}$) | dimensionless | $0.5860 \pm 0.0348$ | $\mathbf{0.6756 \pm 0.0316}$ | $+0.0896$ | $+15.3\%$ |
| | POD | dimensionless | $0.2590 \pm 0.0229$ | $\mathbf{0.4104 \pm 0.0669}$ | $+0.1514$ | $+58.5\%$ |
| | FAR | dimensionless | $\mathbf{0.4725 \pm 0.0341}$ | $0.5639 \pm 0.0193$ | $+0.0914$ | $+19.3\%$ |
| | QPF RMSE | $\text{mm/hr}$ | $\mathbf{1.4404 \pm 0.0116}$ | $1.4927 \pm 0.0673$ | $+0.0523$ | $+3.6\%$ |
| | QPF MAE | $\text{mm/hr}$ | $\mathbf{0.5420 \pm 0.0023}$ | $0.5931 \pm 0.0228$ | $+0.0511$ | $+9.4\%$ |
| | Pearson Correlation ($r$) | dimensionless | $\mathbf{0.6325 \pm 0.0087}$ | $0.6321 \pm 0.0056$ | $-0.0004$ | $-0.1\%$ |
| **+120m** | **Pointwise CSI** | dimensionless | $0.0796 \pm 0.0300$ | $\mathbf{0.1685 \pm 0.0149}$ | $\mathbf{+0.0889}$ | **$+111.7\%$ ($2.12\times$)** |
| | **Circular FSS ($30\text{ km}$)** | dimensionless | $0.3232 \pm 0.1158$ | $\mathbf{0.5560 \pm 0.0307}$ | $\mathbf{+0.2328}$ | **$+72.0\%$** |
| | POD | dimensionless | $0.0883 \pm 0.0380$ | $\mathbf{0.2386 \pm 0.0213}$ | $+0.1503$ | $+170.2\%$ ($2.70\times$) |
| | FAR | dimensionless | $\mathbf{0.5107 \pm 0.0420}$ | $0.6354 \pm 0.0258$ | $+0.1247$ | $+24.4\%$ |
| | QPF RMSE | $\text{mm/hr}$ | $\mathbf{1.7972 \pm 0.0385}$ | $1.8276 \pm 0.0257$ | $+0.0304$ | $+1.7\%$ |
| | QPF MAE | $\text{mm/hr}$ | $\mathbf{0.7118 \pm 0.0181}$ | $0.7782 \pm 0.0084$ | $+0.0664$ | $+9.3\%$ |
| | Pearson Correlation ($r$) | dimensionless | $\mathbf{0.4895 \pm 0.0260}$ | $0.4747 \pm 0.0166$ | $-0.0148$ | $-3.0\%$ |
| **+180m** | **Pointwise CSI** | dimensionless | $0.0183 \pm 0.0107$ | $\mathbf{0.1111 \pm 0.0184}$ | $\mathbf{+0.0928}$ | **$+507.1\%$ ($6.07\times$)** |
| | **Circular FSS ($30\text{ km}$)** | dimensionless | $0.0740 \pm 0.0535$ | $\mathbf{0.4016 \pm 0.0523}$ | $\mathbf{+0.3276}$ | **$+442.7\%$ ($5.43\times$)** |
| | POD | dimensionless | $0.0188 \pm 0.0113$ | $\mathbf{0.1431 \pm 0.0238}$ | $+0.1243$ | $+661.2\%$ ($7.61\times$) |
| | FAR | dimensionless | $\mathbf{0.4924 \pm 0.0675}$ | $0.6690 \pm 0.0372$ | $+0.1766$ | $+35.9\%$ |
| | **QPF RMSE** | $\text{mm/hr}$ | $2.0396 \pm 0.0450$ | $\mathbf{2.0145 \pm 0.0255}$ | $\mathbf{-0.0251}$ | **$-1.2\%$ (Better)** |
| | QPF MAE | $\text{mm/hr}$ | $\mathbf{0.8304 \pm 0.0218}$ | $0.8941 \pm 0.0180$ | $+0.0637$ | $+7.7\%$ |
| | Pearson Correlation ($r$) | dimensionless | $\mathbf{0.4203 \pm 0.0343}$ | $0.3991 \pm 0.0221$ | $-0.0212$ | $-5.0\%$ |
| **+360m** | Pointwise CSI | dimensionless | $0.0000 \pm 0.0000$ | $\mathbf{0.0037 \pm 0.0037}$ | $+0.0037$ | Non-zero restored |
| | Circular FSS ($30\text{ km}$) | dimensionless | $0.0000 \pm 0.0000$ | $\mathbf{0.0193 \pm 0.0172}$ | $+0.0193$ | Non-zero restored |
| | POD | dimensionless | $0.0000 \pm 0.0000$ | $\mathbf{0.0038 \pm 0.0038}$ | $+0.0038$ | Non-zero restored |
| | FAR | dimensionless | $0.0000 \pm 0.0000$ | $0.8306 \pm 0.1708$ | $+0.8306$ | Non-zero restored |
| | **QPF RMSE** | $\text{mm/hr}$ | $2.3842 \pm 0.0101$ | $\mathbf{2.3520 \pm 0.0144}$ | $\mathbf{-0.0322}$ | **$-1.4\%$ (Better)** |
| | QPF MAE | $\text{mm/hr}$ | $\mathbf{1.0923 \pm 0.0080}$ | $1.1242 \pm 0.0161$ | $+0.0319$ | $+2.9\%$ |
| | Pearson Correlation ($r$) | dimensionless | $0.1180 \pm 0.0289$ | $\mathbf{0.1281 \pm 0.0340}$ | $+0.0101$ | $+8.6\%$ |

---

## Table 2: Per-Seed Breakdown and Paired Differences (Validation Split)

Demonstrates exact per-seed replication across random initializations. Data sourced from `PAIRED_SEED_COMPARISON.csv`.

| Horizon | Random Seed | Exp A CSI | Exp D CSI | Paired Gain ($\Delta_{\text{CSI}}$) | Exp A FSS ($30\text{ km}$) | Exp D FSS ($30\text{ km}$) | Paired Gain ($\Delta_{\text{FSS}}$) | Exp A RMSE | Exp D RMSE | Paired Gain ($\Delta_{\text{RMSE}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+30m** | 42 | $0.3203$ | $0.3766$ | $+0.0563$ | $0.7061$ | $0.7890$ | $+0.0829$ | $1.1279$ | $1.1213$ | $-0.0066$ |
| | 123 | $0.3717$ | $0.3999$ | $+0.0282$ | $0.7750$ | $0.7960$ | $+0.0210$ | $1.1044$ | $1.2090$ | $+0.1046$ |
| | 999 | $0.3554$ | $0.3693$ | $+0.0139$ | $0.7939$ | $0.7724$ | $-0.0215$ | $1.1128$ | $1.1291$ | $+0.0163$ |
| | **Mean** | $0.3491$ | $0.3819$ | **$+0.0328$** | $0.7583$ | $0.7858$ | **$+0.0275$** | $1.1150$ | $1.1531$ | $+0.0381$ |
| **+60m** | 42 | $0.2002$ | $0.2676$ | $+0.0674$ | $0.5588$ | $0.6814$ | $+0.1226$ | $1.4272$ | $1.4511$ | $+0.0239$ |
| | 123 | $0.2081$ | $0.2882$ | $+0.0801$ | $0.5743$ | $0.7039$ | $+0.1296$ | $1.4493$ | $1.5703$ | $+0.1210$ |
| | 999 | $0.2200$ | $0.2429$ | $+0.0229$ | $0.6250$ | $0.6414$ | $+0.0164$ | $1.4446$ | $1.4566$ | $+0.0120$ |
| | **Mean** | $0.2094$ | $0.2662$ | **$+0.0568$** | $0.5860$ | $0.6756$ | **$+0.0895$** | $1.4404$ | $1.4927$ | $+0.0523$ |
| **+120m** | 42 | $0.0733$ | $0.1543$ | $+0.0810$ | $0.3000$ | $0.5348$ | $+0.2348$ | $1.7784$ | $1.8385$ | $+0.0601$ |
| | 123 | $0.0532$ | $0.1673$ | $+0.1141$ | $0.2209$ | $0.5419$ | $+0.3210$ | $1.8415$ | $1.8461$ | $+0.0046$ |
| | 999 | $0.1123$ | $0.1840$ | $+0.0717$ | $0.4486$ | $0.5912$ | $+0.1426$ | $1.7718$ | $1.7982$ | $+0.0264$ |
| | **Mean** | $0.0796$ | $0.1685$ | **$+0.0889$** | $0.3232$ | $0.5560$ | **$+0.2328$** | $1.7972$ | $1.8276$ | $+0.0304$ |
| **+180m** | 42 | $0.0135$ | $0.0970$ | $+0.0835$ | $0.0460$ | $0.3688$ | $+0.3228$ | $2.0407$ | $2.0300$ | $-0.0107$ |
| | 123 | $0.0109$ | $0.1045$ | $+0.0936$ | $0.0406$ | $0.3742$ | $+0.3336$ | $2.0840$ | $2.0284$ | $-0.0556$ |
| | 999 | $0.0306$ | $0.1319$ | $+0.1013$ | $0.1355$ | $0.4618$ | $+0.3263$ | $1.9941$ | $1.9850$ | $-0.0091$ |
| | **Mean** | $0.0183$ | $0.1111$ | **$+0.0928$** | $0.0740$ | $0.4016$ | **$+0.3276$** | $2.0396$ | $2.0145$ | **$-0.0251$** |
| **+360m** | 42 | $0.0000$ | $0.0037$ | $+0.0037$ | $0.0000$ | $0.0202$ | $+0.0202$ | $2.3924$ | $2.3381$ | $-0.0543$ |
| | 123 | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0016$ | $+0.0016$ | $2.3873$ | $2.3668$ | $-0.0205$ |
| | 999 | $0.0000$ | $0.0074$ | $+0.0074$ | $0.0000$ | $0.0360$ | $+0.0360$ | $2.3729$ | $2.3510$ | $-0.0219$ |
| | **Mean** | $0.0000$ | $0.0037$ | **$+0.0037$** | $0.0000$ | $0.0193$ | **$+0.0193$** | $2.3842$ | $2.3520$ | **$-0.0322$** |

---

## Table 3: Confirmatory Test Set Generalization (July 12–14, 2023, $N=129$ Sequences)

Out-of-sample benchmark evaluation across 72.0 total observation hours. Evaluated at $\ge 5.0\text{ mm/hr}$ with complete 6.0-hour block decorrelation ($L=12$ sequences).

| Horizon | Metric | Exp A (Benchmark Mean) | Exp D (Benchmark Mean) | Mean Difference ($\bar{\Delta}$) | 95% Bootstrap CI | Continuity-Corrected $p$-value | Significant? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+30m** | Pointwise CSI | $0.5404 \pm 0.0177$ | $\mathbf{0.5469 \pm 0.0145}$ | $+0.0065$ | $[-0.0068, +0.0201]$ | $p = 0.2318$ | NO |
| | Circular FSS ($30\text{ km}$) | $0.8938 \pm 0.0107$ | $\mathbf{0.8970 \pm 0.0154}$ | $+0.0032$ | $[-0.0075, +0.0176]$ | $p = 0.2647$ | NO |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{0.7746 \pm 0.0171}$ | $0.7876 \pm 0.0286$ | $+0.0130$ | $[-0.0019, +0.0284]$ | $p = 0.9481$ | NO |
| **+60m** | Pointwise CSI | $0.3734 \pm 0.0229$ | $\mathbf{0.3910 \pm 0.0019}$ | $+0.0176$ | $[+0.0051, +0.0312]$ | $\mathbf{p = 0.0030}$ | **YES** |
| | Circular FSS ($30\text{ km}$) | $0.7899 \pm 0.0189$ | $\mathbf{0.7986 \pm 0.0178}$ | $+0.0087$ | $[-0.0052, +0.0229]$ | $p = 0.1259$ | NO |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{1.0376 \pm 0.0168}$ | $1.0567 \pm 0.0378$ | $+0.0191$ | $[-0.0048, +0.0435]$ | $p = 0.9161$ | NO |
| **+120m** | **Pointwise CSI** | $0.1233 \pm 0.0413$ | $\mathbf{0.2133 \pm 0.0285}$ | $\mathbf{+0.0901}$ | $[+0.0633, +0.1139]$ | $\mathbf{p < 0.001}$ | **YES** |
| | **Circular FSS ($30\text{ km}$)** | $0.4113 \pm 0.1009$ | $\mathbf{0.6000 \pm 0.0344}$ | $\mathbf{+0.1887}$ | $[+0.1340, +0.2526]$ | $\mathbf{p < 0.001}$ | **YES** |
| | QPF RMSE ($\text{mm/hr}$) | $1.3757 \pm 0.0146$ | $\mathbf{1.3698 \pm 0.0155}$ | $-0.0059$ | $[-0.0398, +0.0289]$ | $p = 0.3546$ | NO |
| **+180m** | **Pointwise CSI** | $0.0120 \pm 0.0132$ | $\mathbf{0.1300 \pm 0.0231}$ | $\mathbf{+0.1180}$ | $[+0.0652, +0.1654]$ | $\mathbf{p < 0.001}$ | **YES** |
| | **Circular FSS ($30\text{ km}$)** | $0.0527 \pm 0.0556$ | $\mathbf{0.4333 \pm 0.0424}$ | $\mathbf{+0.3807}$ | $[+0.2241, +0.4789]$ | $\mathbf{p < 0.001}$ | **YES** |
| | QPF RMSE ($\text{mm/hr}$) | $1.5224 \pm 0.0122$ | $\mathbf{1.5216 \pm 0.0165}$ | $-0.0008$ | $[-0.0617, +0.0545]$ | $p = 0.4490$ | NO |
| **+360m** | Pointwise CSI | $0.0000 \pm 0.0000$ | $\mathbf{0.0125 \pm 0.0129}$ | $+0.0125$ | $[+0.0038, +0.0223]$ | $\mathbf{p = 0.0020}$ | **YES** |
| | Circular FSS ($30\text{ km}$) | $0.0000 \pm 0.0000$ | $\mathbf{0.0526 \pm 0.0495}$ | $+0.0526$ | $[+0.0126, +0.0984]$ | $\mathbf{p < 0.001}$ | **YES** |
| | QPF RMSE ($\text{mm/hr}$) | $\mathbf{1.6727 \pm 0.0056}$ | $1.6771 \pm 0.0475$ | $+0.0044$ | $[-0.0315, +0.0446]$ | $p = 0.5844$ | NO |

---

## Table 4: Spatial Verification Scale Decomposition (Validation Split)

Fractions Skill Score evaluated across multiple circular Euclidean neighborhood radii ($k \in \{1, 2, 3, 5\}$ cells) at the $\ge 5.0\text{ mm/hr}$ convective threshold. Data sourced from `SPATIAL_VERIFICATION_RESULTS.csv`.

| Horizon | Neighborhood Radius | Physical Window Size | Baseline (Exp A Mean) | Combined (Exp D Mean) | Absolute FSS Gain | Theoretical Ref ($0.5 + f_0/2$) | Skill Characterization |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **+30m** | $10\text{ km}$ ($k=1$) | 5 cells | $0.6015 \pm 0.0289$ | $\mathbf{0.6385 \pm 0.0201}$ | $+0.0370$ | $\sim 0.52$ | Exceeds theoretical ref |
| | $20\text{ km}$ ($k=2$) | 13 cells | $0.6974 \pm 0.0376$ | $\mathbf{0.7301 \pm 0.0152}$ | $+0.0327$ | $\sim 0.52$ | Exceeds theoretical ref |
| | $30\text{ km}$ ($k=3$) | 29 cells | $0.7583 \pm 0.0464$ | $\mathbf{0.7858 \pm 0.0121}$ | $+0.0275$ | $\sim 0.52$ | Exceeds theoretical ref |
| | $50\text{ km}$ ($k=5$) | 81 cells | $0.8251 \pm 0.0504$ | $\mathbf{0.8462 \pm 0.0094}$ | $+0.0211$ | $\sim 0.52$ | Exceeds theoretical ref |
| **+60m** | $10\text{ km}$ ($k=1$) | 5 cells | $0.3957 \pm 0.0185$ | $\mathbf{0.4851 \pm 0.0345}$ | $+0.0894$ | $\sim 0.52$ | Approaching theoretical ref |
| | $20\text{ km}$ ($k=2$) | 13 cells | $0.5113 \pm 0.0267$ | $\mathbf{0.6025 \pm 0.0340}$ | $+0.0912$ | $\sim 0.52$ | Exceeds theoretical ref |
| | $30\text{ km}$ ($k=3$) | 29 cells | $0.5860 \pm 0.0348$ | $\mathbf{0.6756 \pm 0.0316}$ | $+0.0896$ | $\sim 0.52$ | Exceeds theoretical ref |
| | $50\text{ km}$ ($k=5$) | 81 cells | $0.6793 \pm 0.0425$ | $\mathbf{0.7592 \pm 0.0272}$ | $+0.0799$ | $\sim 0.52$ | Exceeds theoretical ref |
| **+120m** | $10\text{ km}$ ($k=1$) | 5 cells | $0.1654 \pm 0.0620$ | $\mathbf{0.3448 \pm 0.0253}$ | $\mathbf{+0.1794}$ | $\sim 0.52$ | Below theoretical ref |
| | $20\text{ km}$ ($k=2$) | 13 cells | $0.2570 \pm 0.0934$ | $\mathbf{0.4727 \pm 0.0294}$ | $\mathbf{+0.2157}$ | $\sim 0.52$ | Approaching theoretical ref |
| | $30\text{ km}$ ($k=3$) | 29 cells | $0.3232 \pm 0.1158$ | $\mathbf{0.5560 \pm 0.0307}$ | $\mathbf{+0.2328}$ | $\sim 0.52$ | Exceeds theoretical ref |
| | $50\text{ km}$ ($k=5$) | 81 cells | $0.4190 \pm 0.1388$ | $\mathbf{0.6558 \pm 0.0316}$ | $\mathbf{+0.2368}$ | $\sim 0.52$ | Exceeds theoretical ref |
| **+180m** | $10\text{ km}$ ($k=1$) | 5 cells | $0.0308 \pm 0.0205$ | $\mathbf{0.2162 \pm 0.0329}$ | $\mathbf{+0.1854}$ | $\sim 0.52$ | Below theoretical ref |
| | $20\text{ km}$ ($k=2$) | 13 cells | $0.0543 \pm 0.0384$ | $\mathbf{0.3298 \pm 0.0454}$ | $\mathbf{+0.2755}$ | $\sim 0.52$ | Below theoretical ref |
| | $30\text{ km}$ ($k=3$) | 29 cells | $0.0740 \pm 0.0535$ | $\mathbf{0.4016 \pm 0.0523}$ | $\mathbf{+0.3276}$ | $\sim 0.52$ | Approaching theoretical ref |
| | $50\text{ km}$ ($k=5$) | 81 cells | $0.1068 \pm 0.0768$ | $\mathbf{0.4990 \pm 0.0594}$ | $\mathbf{+0.3922}$ | $\sim 0.52$ | Approaching theoretical ref |
| **+360m** | $10\text{ km}$ ($k=1$) | 5 cells | $0.0000 \pm 0.0000$ | $\mathbf{0.0076 \pm 0.0069}$ | $+0.0076$ | $\sim 0.53$ | Weak |
| | $20\text{ km}$ ($k=2$) | 13 cells | $0.0000 \pm 0.0000$ | $\mathbf{0.0143 \pm 0.0129}$ | $+0.0143$ | $\sim 0.53$ | Weak |
| | $30\text{ km}$ ($k=3$) | 29 cells | $0.0000 \pm 0.0000$ | $\mathbf{0.0193 \pm 0.0172}$ | $+0.0193$ | $\sim 0.53$ | Weak |
| | $50\text{ km}$ ($k=5$) | 81 cells | $0.0000 \pm 0.0000$ | $\mathbf{0.0270 \pm 0.0238}$ | $+0.0270$ | $\sim 0.53$ | Weak |

*Note on Threshold:* The value $0.5 + f_0/2 \approx 0.52$ represents the theoretical heuristic defined by Roberts & Lean (2008), where forecast skill exceeds that of a random forecast with the same event frequency. Because no formal operational standard was pre-registered in the project synopsis, this value is provided solely as a theoretical reference for comparative interpretation rather than an operational certification.

---

## Table 5: Intensity Distribution and Core Displacement Diagnostics

Evaluated across July 5, 2025 validation episode. Observed maximum intensity was $36.07\text{ mm/hr}$ (+30m, +60m) and $48.58\text{ mm/hr}$ (+120m, +180m, +360m). Observed 99th percentile was $8.24\text{--}10.63\text{ mm/hr}$. Data sourced from `PREDICTION_INTENSITY_DIAGNOSTICS.csv`.

| Horizon | Configuration | Maximum Intensity ($\text{mm/hr}$) | 95th Percentile ($\text{mm/hr}$) | 99th Percentile ($\text{mm/hr}$) | Cells $\ge 5\text{ mm/hr}$ | Cells $\ge 10\text{ mm/hr}$ | Frequency Bias ($B$) | Matched Centroid Displacement |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+30m** | Observed Field | $36.07$ | $3.74$ | $8.24$ | $955$ | $191$ | $1.000$ | Baseline ($0.0\text{ km}$) |
| | Baseline (Exp A) | $18.12 \pm 1.83$ | $3.09 \pm 0.19$ | $6.73 \pm 0.46$ | $686 \pm 113$ | $90 \pm 18$ | $0.719 \pm 0.118$ | $69.9 \pm 3.0\text{ km}$ |
| | Combined (Exp D) | $\mathbf{22.13 \pm 2.24}$ | $\mathbf{3.60 \pm 0.39}$ | $\mathbf{8.18 \pm 1.01}$ | $\mathbf{997 \pm 181}$ | $\mathbf{180 \pm 75}$ | $\mathbf{1.044 \pm 0.190}$ | $68.5 \pm 14.2\text{ km}$ |
| **+60m** | Observed Field | $36.07$ | $4.04$ | $8.88$ | $1,116$ | $233$ | $1.000$ | Baseline ($0.0\text{ km}$) |
| | Baseline (Exp A) | $16.37 \pm 1.49$ | $2.74 \pm 0.19$ | $6.14 \pm 0.42$ | $551 \pm 82$ | $60 \pm 25$ | $0.494 \pm 0.074$ | $84.4 \pm 3.3\text{ km}$ |
| | Combined (Exp D) | $\mathbf{16.88 \pm 4.03}$ | $\mathbf{3.81 \pm 0.41}$ | $\mathbf{8.00 \pm 1.21}$ | $\mathbf{1,055 \pm 213}$ | $\mathbf{150 \pm 106}$ | $\mathbf{0.946 \pm 0.191}$ | $73.0 \pm 10.6\text{ km}$ |
| **+120m** | Observed Field | $48.58$ | $4.55$ | $9.73$ | $1,366$ | $328$ | $1.000$ | Baseline ($0.0\text{ km}$) |
| | Baseline (Exp A) | $11.22 \pm 0.34$ | $2.09 \pm 0.38$ | $4.50 \pm 0.73$ | $252 \pm 127$ | $8 \pm 2$ | $0.184 \pm 0.093$ | $108.6 \pm 19.7\text{ km}$ |
| | Combined (Exp D) | $\mathbf{14.70 \pm 0.99}$ | $\mathbf{3.64 \pm 0.24}$ | $\mathbf{6.93 \pm 0.22}$ | $\mathbf{895 \pm 57}$ | $\mathbf{55 \pm 42}$ | $\mathbf{0.655 \pm 0.042}$ | $99.6 \pm 10.0\text{ km}$ |
| **+180m** | Observed Field | $48.58$ | $4.85$ | $10.31$ | $1,544$ | $385$ | $1.000$ | Baseline ($0.0\text{ km}$) |
| | Baseline (Exp A) | $7.06 \pm 0.06$ | $1.49 \pm 0.35$ | $3.05 \pm 0.65$ | $60 \pm 42$ | $0 \pm 0$ | $0.038 \pm 0.027$ | $186.2 \pm 66.0\text{ km}$ |
| | Combined (Exp D) | $\mathbf{13.15 \pm 1.87}$ | $\mathbf{3.28 \pm 0.21}$ | $\mathbf{6.17 \pm 0.12}$ | $\mathbf{666 \pm 46}$ | $\mathbf{26 \pm 19}$ | $\mathbf{0.431 \pm 0.030}$ | $\mathbf{110.1 \pm 0.3\text{ km}}$ |
| **+360m** | Observed Field | $48.58$ | $5.29$ | $10.63$ | $1,858$ | $471$ | $1.000$ | Baseline ($0.0\text{ km}$) |
| | Baseline (Exp A) | $1.75 \pm 0.39$ | $0.77 \pm 0.13$ | $1.04 \pm 0.17$ | $0 \pm 0$ | $0 \pm 0$ | $0.000 \pm 0.000$ | Skipped ($0$ events) |
| | Combined (Exp D) | $\mathbf{7.47 \pm 0.89}$ | $\mathbf{1.48 \pm 0.04}$ | $\mathbf{3.06 \pm 0.11}$ | $\mathbf{44 \pm 4}$ | $0 \pm 0$ | $\mathbf{0.024 \pm 0.002}$ | $167.9 \pm 57.0\text{ km}$ |
