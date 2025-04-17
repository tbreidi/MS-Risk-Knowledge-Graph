# MS-Risk-Knowledge-Graph

A comprehensive pipeline for building, integrating, and analyzing a knowledge graph (KG) that combines Social Determinants of Health (SDOH) with biomedical data for Multiple Sclerosis (MS) risk analysis, utilizing Graph Neural Networks (GNN) for link prediction.

---

## 📖 Overview

This repository implements a structured workflow to:

- Extract and preprocess biomedical and social health data from PubMed abstracts.
- Perform hybrid entity recognition (PubMedBERT, SpaCy, LLM refinement).
- Map entities to standardized ontologies (UMLS, SNOMED CT).
- Extract and classify relationships using transformers (BioBERT) and co-occurrence statistics.
- Assemble a heterogeneous knowledge graph and integrate it with PrimeKG's MS subgraph.
- Perform link prediction using GraphSAGE, achieving an AUC of 0.91.

The final graph includes ~33,250 nodes and ~139,100 edges, providing novel insights into MS risk factors.

---

## 📁 Repository Structure
```
.
├── data/
│   ├── createDataset.py
│   ├── getJournals.py
│   └── ExploratoryAnalysis.py
│
├── analysis/
│   ├── NERLabelling.py
│   ├── NERMapping.py
│   ├── RelationshipExtractionCooccurrence.py
│   ├── RelationshipExtractionSemantic.py
│   └── PerformanceEstimation.py
│
├── ontology/
│   ├── SDOH_RISK_SNOMEDCT.py
│   └── SDOH_Risk_CorrelationPoC.py
│
├── PrimeKGIntegration/
│   └── MergePrimeKG.py
│
├── graphNN/
│   ├── KG_creation.py
│   └── LinkPrediction.py
│
├── charts/
│   └── ProjectTimeline.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

Clone the repository:
```bash
git clone https://github.com/tbreidi/MS-Risk-Knowledge-Graph.git
cd MS-Risk-Knowledge-Graph
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Pipeline Workflow

### 1. Data Collection & Preprocessing
- Defined query in `data/createDataset.py`:
```
"multiple sclerosis [MESH]" AND ("prevention" OR "social determinants" OR "environmental factors" OR "lifestyle factors" OR "modifiable risk factors")
```
- Preprocess and explore abstracts:
```bash
python data/ExploratoryAnalysis.py
```

### 2. Entity Recognition & Mapping
- Hybrid NER using PubMedBERT (biomedical), SpaCy + WHO seed terms (SDOH), and LLM refinement:
```bash
python analysis/NERLabelling.py --input data/pubmed_abstracts.csv --output analysis/entities.csv
python analysis/NERMapping.py --input analysis/entities.csv --output analysis/mapped_entities.csv
```

### 3. Relationship Extraction
- Co-occurrence scoring and transformer-based semantic classification:
```bash
python analysis/RelationshipExtractionCooccurrence.py --entities analysis/mapped_entities.csv --output analysis/cooc_edges.csv
python analysis/RelationshipExtractionSemantic.py --entities analysis/mapped_entities.csv --output analysis/semantic_edges.csv
```
- Merge and threshold using `analysis/PerformanceEstimation.py`.

### 4. Ontological Correlation Analysis
- SDOH to Risk factors via SNOMED CT:
```bash
python ontology/SDOH_RISK_SNOMEDCT.py
```
- Correlation analysis using CDC SVI data:
```bash
python ontology/SDOH_Risk_CorrelationPoC.py --input ontology/svi-pub-data.csv
```

### 5. Knowledge Graph Assembly
- Assemble the graph with NetworkX:
```bash
python graphNN/KG_creation.py
```

### 6. PrimeKG Integration
- Merge with PrimeKG’s MS-relevant subgraph:
```bash
python PrimeKGIntegration/MergePrimeKG.py \
  --custom analysis/combined_edges.csv \
  --prime data/primekg_triples.csv \
  --disease_map data/mesh_omim_to_mondo.csv \
  --drug_map data/mesh_to_drugbank.csv \
  --ms_id MONDO:0005301 \
  --output merged_triples.csv
```

### 7. Link Prediction via GNN
- Train/test split, node embeddings (node2vec → GraphSAGE), and prediction:
```bash
python graphNN/LinkPrediction.py --edges merged_triples.csv --embeddings node2vec.emb --output predictions.csv
```

---

## 📌 Usage Examples

Run the entire pipeline

---

## 📊 Visualization & Reporting

---

## 🤝 Contributing

- Fork the repository.
- Create a branch: `git checkout -b feature/YourFeature`.
- Commit your changes.
- Open a Pull Request.

---

## 📜 License & Citation

License: MIT

## 📧 Contact

Tim Breitenfelder  
[tim.breitenfelder@kellogg.ox.ac.uk](mailto:tim.breitenfelder@kellogg.ox.ac.uk)