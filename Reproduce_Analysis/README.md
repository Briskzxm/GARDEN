# Reproducible Analysis Pipeline

We provide a modular suite of Python scripts to perform essential downstream analyses in spatial transcriptomics. Each script implements a specific analytical workflow and can be executed independently with minimal configuration.

---
### Analysis Scripts

#### **`Cell-network.R`** – Cell Network Visualization
  
Visualizes spatial proximity relationships between two cell types using **MERINGUE** to reveal structured interaction patterns across the tissue.

#### **`Coappear.py`** – Co-occurrence probability Visualization
  
Computes co-occurrence probabilities between different clusters using **squidpy**, quantifying spatial associations and enrichment patterns that reflect coordinated cellular organization within the tissue microenvironment.

#### **`LR-analysis.py`** – Receptor-ligand Analysis
  
Receptor-ligand analysis in spatial transcriptomics dataset using **squidpy**.


#### **`Deconvolution.py`** – Spot Decomposition  
Performs cell-type deconvolution on spatial transcriptomics data using **GraphST**, recovering single-cell resolution identities within each capture spot.

#### **`Tangram.py`** – Mapping Data
Mapping single cell data on spatial data using the **Tangram** method where is suitable for single-cell resolution spatial transcriptome data.


#### **`Cell_Cell_Communication.py`** – Inter-cluster Communication  
Analyzes cell–cell communication networks between spatial clusters using **CellChat**, identifying significant ligand–receptor interactions and characterizing communication patterns across tissue regions.

#### **`Trajectory.py`** – Spatial Trajectory Inference  
Infers developmental or state-transition trajectories within spatially resolved data using **VIA**, reconstructing pseudotemporal dynamics across tissue architectures.

####  **`Hotspot.py`** – Spatial Pattern Detection  
Identifies spatially coherent gene modules and expression hotspots using **Stereopy**, revealing localized functional domains and spatially variable genes within the tissue.

####  **`Spateo-3D.py`** – 3D Visulization and  Reconstruction 
Reconstructs and visualizes tissue architectures in 3D using **Spateo**, revealing the three-dimensional morphological organization of distinct cell types and enabling detailed visualization of cell shape, spatial arrangement, and structural heterogeneity within the tissue.

####  **`CSI-heatmap.py`** – Heatmap Visulization
Visualizes the Connection Specificity Index (CSI) between different transcription factors as a heatmap, highlighting specificity-driven regulatory relationships and revealing coordinated or distinct TF interaction patterns.

####  **`Correlation.R`** – correlation Visulization
Calculates the correlation between the proportions of spots occupied by the two cell types, quantifying their spatial co-distribution and revealing potential coordinated localization or mutual exclusivity across the tissue landscape.

--- 
### Environment Setup  
To avoid dependency conflicts, we recommend creating separate conda environments for each analytical module. Required tools and their source repositories are listed below.

---

### Tool References  
- **MERINGUE** - Cell network: [https://github.com/JEFworks-Lab/MERINGUE](https://github.com/JEFworks-Lab/MERINGUE)  


- **squidpy** - Spatial analysis: [https://github.com/scverse/squidpy](https://github.com/scverse/squidpy)  

- **Spateo** – 3D framework: [https://github.com/aristoteleo/spateo-release](https://github.com/aristoteleo/spateo-release) 

- **GraphST** – Spot deconvolution: [https://github.com/JinmiaoChenLab/GraphST](https://github.com/JinmiaoChenLab/GraphST)  

- **Tangram** – Cell mapping: [https://github.com/broadinstitute/Tangram](https://github.com/broadinstitute/Tangram) 

- **OmicVerse** – Core computational framework: [https://github.com/Starlitnightly/omicverse](https://github.com/Starlitnightly/omicverse)  

- **CellChat** – Cell–cell communication analysis: [https://github.com/sqjin/CellChat](https://github.com/sqjin/CellChat)  

- **Stereopy** – Spatial pattern detection: [https://github.com/STOmics/Stereopy](https://github.com/STOmics/Stereopy)

---
### Reproducible Demo Data Reference
- Breast Cancer - https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast