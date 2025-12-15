# Reproducible Analysis Pipeline

We provide a modular suite of Python scripts to perform essential downstream analyses in spatial transcriptomics. Each script implements a specific analytical workflow and can be executed independently with minimal configuration.

---
### Analysis Scripts

#### **`Deconvolution.py`** – Spot Decomposition  
Performs cell-type deconvolution on spatial transcriptomics data using **GraphST**, recovering single-cell resolution identities within each capture spot.

#### **`Cell_Cell_Communication.py`** – Inter-cluster Communication  
Analyzes cell–cell communication networks between spatial clusters using **CellChat**, identifying significant ligand–receptor interactions and characterizing communication patterns across tissue regions.

#### **`Trajectory.py`** – Spatial Trajectory Inference  
Infers developmental or state-transition trajectories within spatially resolved data using **VIA**, reconstructing pseudotemporal dynamics across tissue architectures.

####  **`Hotspot.py`** – Spatial Pattern Detection  
Identifies spatially coherent gene modules and expression hotspots using **Stereopy**, revealing localized functional domains and spatially variable genes within the tissue.

--- 
### Environment Setup  
To avoid dependency conflicts, we recommend creating separate conda environments for each analytical module. Required tools and their source repositories are listed below.

---

### Tool References  
- **GraphST** – Spot deconvolution: [https://github.com/JinmiaoChenLab/GraphST](https://github.com/JinmiaoChenLab/GraphST)  

- **OmicVerse** – Core computational framework: [https://github.com/Starlitnightly/omicverse](https://github.com/Starlitnightly/omicverse)  

- **CellChat** – Cell–cell communication analysis: [https://github.com/sqjin/CellChat](https://github.com/sqjin/CellChat)  

- **Stereopy** – Spatial pattern detection: [https://github.com/STOmics/Stereopy](https://github.com/STOmics/Stereopy)

---
### Reproducible Demo Data Reference
- Breast Cancer - https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast