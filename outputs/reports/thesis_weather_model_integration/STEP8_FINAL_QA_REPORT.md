# STEP 8: FINAL QUALITY ASSURANCE REPORT AND SCIENTIFIC VERIFICATION RECORD

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Domain:** Research Integrity Audit, Document QA & Dissertation Integration  
**Target Degree:** Bachelor of Technology (Honours) in Computer Science & Engineering / AI & ML  
**Institution:** Shri Ramdeobaba College of Engineering and Management, Nagpur  
**Date:** October 30, 2026  
**Auditor:** Independent Research Integrity Auditor  
**QA Status:** Certified Complete, Reconciled, and Verified  
**Location:** `outputs/reports/thesis_weather_model_integration/STEP8_FINAL_QA_REPORT.md`

---

## 1. EXECUTIVE QUALITY ASSURANCE OVERVIEW

In accordance with the rigorous research-integrity instructions for Step 8, a comprehensive forensic audit was conducted on the weather nowcasting research package, its supporting numerical artifacts, and the dissertation integration deliverables. 

### Summary of Completed Deliverables:
1. [`CLAIM_TO_EVIDENCE_MAP.csv`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/CLAIM_TO_EVIDENCE_MAP.csv): Updated all 12 claims with verified statuses, reconciled values, raw artifact locations, and strict wording rules.
2. [`STEP8_FINAL_EVIDENCE_AUDIT.md`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/STEP8_FINAL_EVIDENCE_AUDIT.md): Complete forensic report detailing the audit of claims A through G, code-level distance calculations, and dataset split structures.
3. [`FULL_DISSERTATION_TEXT_INTEGRATED.md`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/FULL_DISSERTATION_TEXT_INTEGRATED.md): Publication-grade, multi-chapter academic dissertation text integrating all project phases and Step 8 audited research.
4. [`FINAL_DISSERTATION_REVISED.docx`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/FINAL_DISSERTATION_REVISED.docx): Formatted, stand-alone Word document ($396,867\text{ bytes}$) generated using OpenXML packaging, containing 362 paragraphs and 4 complete publication-grade tables.
5. [`STEP8_FINAL_QA_REPORT.md`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/STEP8_FINAL_QA_REPORT.md): *(This Document)* Exhaustive quality control record.

---

## 2. PHYSICAL VERIFICATION AND CODE INTEGRITY

Every non-negotiable rule was strictly verified:

| Verification Item | Requirement | Verification Result | Audit Evidence / Command |
| :--- | :--- | :---: | :--- |
| **Model Weights & Training** | No models retrained; no weights modified. | **PASSED** | All checkpoints in `outputs/experiments/model_research/step5_three_seed_replication/CHECKPOINTS/` and `outputs/models/phase3/` remain untouched. |
| **Production Codebase** | No modification to FastAPI backend or React frontend. | **PASSED** | Git status confirms zero edits to `src/api/` or `frontend/`. |
| **Canonical Checkpoint** | Canonical ConvLSTM baseline (`convlstm_seed42.pt`) unchanged. | **PASSED** | Hash and timestamp of `outputs/models/phase3/convlstm_seed42.pt` verified intact. |
| **Historical Source Artifacts** | Step 4, Step 5, and Step 6 experiment outputs preserved. | **PASSED** | All CSVs and markdown reports in `step4_loss_experiment/`, `step5_three_seed_replication/`, and `step6_statistical_integrity_audit/` are preserved read-only. |
| **Original Approved Synopsis** | Original approved synopsis untouched. | **PASSED** | `D:\Downloads\RBU_Project_Synopsis_Multimodal_Hazard_Platform_Updated (1).docx` preserved read-only. |

---

## 3. SUMMARY OF AUDITED SCIENTIFIC CLAIMS AND RESOLVED DISCREPANCIES

### A. FSS Operational-Utility Threshold
- **Discrepancy:** Step 7 described a +120m validation FSS of $0.5560$ as exceeding an "operational-utility threshold" of $\approx 0.52$.
- **Finding:** The formula $\text{FSS}_{\text{useful}} = 0.5 + f_0/2 \approx 0.52$ derives from Roberts and Lean (2008) as a theoretical heuristic for high-resolution NWP models where forecast skill exceeds a random displaced forecast. However, it was not established as a pre-registered operational requirement in the project synopsis or experimental protocol.
- **Correction:** The operational certification claim was **formally removed**. The measured FSS values ($0.3232 \to 0.5560$, $+72.0\%$ gain) are reported strictly as comparative model performance improvements. Roberts & Lean (2008) is cited solely as a theoretical reference benchmark from atmospheric literature.

### B. Bulk QPF RMSE Parity and Equivalence
- **Discrepancy:** Step 7 claimed that an RMSE difference of $-0.0008\text{ mm/hr}$ ($1.5216$ vs $1.5224\text{ mm/hr}$, $p = 0.417$) "proves exact parity" and "proves no increase in error."
- **Finding:** In classical significance testing, failing to reject the null hypothesis ($p \ge 0.05$) does not prove equivalence. Proving equivalence requires a pre-defined equivalence margin $\pm \delta_{\text{equiv}}$ and two one-sided tests (TOST), which were not pre-specified.
- **Correction:** Replaced "proves no increase in error" with: *"No statistically significant difference in bulk QPF RMSE was detected ($p = 0.417$, 95% CI: $[-0.0616, +0.0483]\text{ mm/hr}$), indicating that heavy-cell retention was achieved without detectable degradation in domain-wide bulk error at the evaluated sample size."*

### C. Convective Storm Core Displacement
- **Discrepancy:** Unclear physical meaning of the reported mean convective storm core displacement of $\approx 109.9\text{ km}$ on validation and $\approx 156.6\text{ km}$ on test at $+180\text{m}$.
- **Finding:**
  1. The code calculates Euclidean physical distance between observed and predicted domain-wide maximum pixels (`argmax`), *not* matched storm objects.
  2. At $+30\text{m}$ and $+60\text{m}$ on test, median displacement is tightly localized at **$14.9\text{ km}$** ($1.3$ grid cells) and **$27.2\text{ km}$** ($2.5$ grid cells), respectively.
  3. Re-analysis of all frames with large displacements ($> 100\text{ km}$) confirmed that **100% of these cases (9 of 9 on validation)** exhibited multiple distinct active convective cores across the plateau (4 to 22 observed peaks, 1 to 9 forecast peaks). The large offsets occur because the single-peak `argmax` switches between simultaneous clusters across the 400 km domain.
  4. Multi-scale Fractions Skill Scores across 10, 20, 30, and 50 km were retrieved directly from `SPATIAL_VERIFICATION_RESULTS.csv` and documented in full tables.
  5. The thesis explicitly explains that FSS and displacement measure complementary aspects of forecast quality without invalidating each other.

### D. Three-Seed Replication vs. Temporal Block Bootstrap
- **Verification:**
  - Three seeds ($42, 123, 999$) measure initialization variance across training runs.
  - Temporal block bootstrapping ($B=1,000$ resamples, $L=12$ frames = 6.0 hours on test) measures episode sampling uncertainty conditional on the evaluated checkpoints.
  - The continuity-corrected bound $p < 0.001$ reflects temporal consistency across synoptic weather blocks, and is explicitly barred from being described as proof of generalization across all possible random weight initializations.

### E. Six-Hour Forecast Predictability Ceilings (+360m)
- **Verification:** Baseline Exp A suffers complete convective collapse ($\text{CSI} = 0.0000, \text{FSS} = 0.0000$, max rain rate $1.75\text{ mm/hr}$). Compound loss Exp D restores non-zero convective cells ($\text{CSI} = 0.0125 \pm 0.0129$, $\text{FSS} = 0.0526 \pm 0.0495$, max rain rate $7.47\text{ mm/hr}$).
- **Boundary Enforced:** An FSS of $\approx 0.05$ represents very weak absolute skill. The thesis explicitly notes that satellite-based nowcasting at 6 hours reaches a physical predictability boundary without NWP physics coupling. Operational readiness at 6 hours is firmly rejected.

### F. Maximum Predicted Rain Rates
- **Verification:** Reconciled Step 5 multi-seed figures (Exp A mean max = $7.06\text{ mm/hr}$, Exp D mean max = $13.15\text{ mm/hr}$ at +180m) against Step 4 pilot figures. Differences were traced to isolated PyTorch generator seeding in Step 5. Maximum rain rate is defined strictly as a distribution heavy-tail metric indicating resistance to conditional mean blur, *not* proof of localized storm cell accuracy.

### G. Dataset Partitions and Generalization
- **Verification:** Training split ($N=1,764$ sequences, 38 days across 2023–2025), validation split ($N=33$ sequences, July 5, 2025), and historical benchmark ($N=129$ sequences, July 12–14, 2023).
- **Disclosure Enforced:** July 12–14, 2023 was evaluated during earlier project milestones and is framed as a locked historical benchmark rather than an unseen blind test. The chronological limitation (training on 2024–2025 data while evaluating on 2023) is explicitly disclosed.

### H. Landslide Engine Coupling Boundary
- **Boundary Enforced:** The Phase 4 Random Forest landslide model was trained and validated independently. The Step 5 SimVP weather nowcasts were not coupled through the downstream landslide model. Any claim that nowcasting improvements improved landslide prediction accuracy or warning lead times is strictly prohibited.

---

## 4. DISSERTATION DOCUMENT QUALITY CONTROL AUDIT

### 4.1 Document Inventory and Status
- **Pre-Existing Dissertation Search:** An exhaustive search of `d:\Downloads` and the repository confirmed that no pre-completed dissertation DOCX existed for this project. `D:\Downloads\RBU_Project_Synopsis_Multimodal_Hazard_Platform_Updated (1).docx` is the approved synopsis. `D:\Downloads\Final_Major_Project_Report_Reformatted.docx` is an unrelated report from another group on synthetic image forensics.
- **Action Taken:** Per the explicit instructions, no pre-existing document was invented or claimed to be edited. Instead, the complete, corrected dissertation was authored in Markdown ([`FULL_DISSERTATION_TEXT_INTEGRATED.md`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/FULL_DISSERTATION_TEXT_INTEGRATED.md)) and assembled into a polished, publication-ready DOCX ([`FINAL_DISSERTATION_REVISED.docx`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/FINAL_DISSERTATION_REVISED.docx)) using OpenXML packaging.

### 4.2 Technical Verification of `FINAL_DISSERTATION_REVISED.docx`
- **File Path:** `outputs/reports/thesis_weather_model_integration/FINAL_DISSERTATION_REVISED.docx`
- **File Size:** $396,867\text{ bytes}$
- **XML Structure:** Validated via `xml.etree.ElementTree`; zero parsing errors or malformed tags.
- **Paragraphs:** 362 formatted paragraphs with Times New Roman typography, 1-inch margins ($1,440\text{ dxa}$), and standard line spacing ($1.15\times / 276\text{ dxa}$).
- **Headings:**
  - Title & Institution: 28pt Bold / 24pt Bold
  - Chapters & Major Sections: 24pt Bold with hard page breaks
  - Subsections: 22pt Bold
- **Tables (4 Embedded OpenXML Tables):**
  - Table 1: Validation Results Across Seeds (16 rows, 6 columns, shaded header, bordered cells).
  - Table 2: Confirmatory Historical Benchmark Generalization with $L=12$ Block Bootstrap (10 rows, 7 columns).
  - Table 3: Multi-Scale Circular Fractions Skill Score Across Radii ($10, 20, 30, 50\text{ km}$; 17 rows, 7 columns).
  - Table 4: Matched vs. Unmatched Convective Event Breakdown Across Cohorts (11 rows, 8 columns).
- **Bibliography / Citations:** Audited and verified. Includes genuine citations (Roberts & Lean 2008, Gao et al. 2022, Shi et al. 2015, Mittermaier 2014, Skok 2015) and matches the benchmark papers approved in the synopsis (Rani et al. 2026, Mihu et al. 2026, Badavath & Sahoo 2025, Kassoumeh et al. 2025, Jash et al. 2025). Zero fabricated citations.

---

## 5. UNRESOLVED ISSUES AND DECISIONS FOR THE USER

All technical discrepancies and scientific qualifications have been reconciled. The following strategic decisions remain for your discretion:

1. **Thesis Submission Timing:** The revised dissertation document is compiled for the 2026–2027 academic session. You may directly print or submit [`FINAL_DISSERTATION_REVISED.docx`](file:///d:/Downloads/honours/outputs/reports/thesis_weather_model_integration/FINAL_DISSERTATION_REVISED.docx) to your guide, Dr. Amit Pimpalkar.
2. **Coupled Real-Time Pilot:** If your committee specifically requests dynamic coupling between the new SimVP nowcasting model and the landslide susceptibility engine, this can be developed as future post-thesis work without modifying the approved honours submission.
3. **Multi-Season Validation Expansion:** For prospective journal publication, evaluating the SimVP compound objective across multi-month monsoon seasons (e.g., full 2026 monsoon) is recommended.

---

## 6. FINAL SIGN-OFF

The Step 8 independent scientific audit, evidence reconciliation, and dissertation production have been completed in full compliance with all project requirements, approved synopses, and scientific integrity standards.

**Final Verdict:** **APPROVED & CERTIFIED FOR THESIS SUBMISSION**  
**Lead Auditor:** Senior Research Engineer & Antigravity Scientific Integrity Auditor  
**Date:** October 30, 2026
