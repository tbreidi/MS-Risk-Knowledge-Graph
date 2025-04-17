import pandas as pd

# --- Configuration ---

# SDOH subcategories to look for (all in lowercase for matching)
SDOH_TERMS = [
    "employment",
    "food insecurity",
    "housing instability",
    "poverty",
    "early childhood development and education",
    "enrollment in higher education",
    "high school graduation",
    "language and literacy",
    "access to health services",
    "access to primary care",
    "health literacy",
    "access to foods that support healthy dietary patterns",
    "crime and violence",
    "environmental conditions",
    "quality of housing",
    "civic participation",
    "discrimination",
    "incarceration",
    "social cohesion"
]

# Risk factors to look for (all in lowercase for matching)
RISK_TERMS = [
    "hla-drb1*15:01 allele",
    "other hla risk alleles",
    "il2ra gene variants",
    "il7r gene variants",
    "cd58 gene variants",
    "epstein–barr virus infection",
    "hhv-6 infection",
    "cytomegalovirus infection",
    "vitamin d deficiency",
    "low sunlight exposure",
    "smoking",  # active or passive
    "exposure to organic solvents",
    "exposure to pesticides",
    "exposure to industrial chemicals and heavy metals",
    "air pollution",
    "high salt intake",
    "high saturated fat consumption",
    "low omega-3 fatty acid intake",
    "low antioxidant intake",
    "micronutrient deficiencies",
    "female gender",
    "hormonal fluctuations",
    "exogenous hormone use",
    "adolescent obesity",
    "high bmi in early life",
    "chronic psychological stress",
    "sedentary lifestyle",
    "circadian disruption",
    "altered gut microbiota",
    "childhood infections",
    "reduced microbial exposure",
    "maternal infections during pregnancy",
    "socioeconomic factors",
    "urban living factors",
    "occupational exposures",
    "epigenetic modifications"
]

# --- Step 1: Read the dataset ---
# The dataset is assumed to be a CSV file with columns:
# id, Title, Authors, Keywords, Published Year, Journal, DOI, States, Cities, Geography Keywords, Citation
df = pd.read_csv("svi-pub-data.csv", encoding="ISO-8859-1")

# Check that the dataset contains the "Keywords" column
if "Keywords" not in df.columns:
    raise ValueError("The dataset must contain a 'Keywords' column.")

# --- Step 2: Create Binary Indicator Columns from Keywords ---
def tokenize_keywords(keywords_str):
    """
    Tokenize a keywords string by splitting on semicolons and stripping whitespace.
    Returns a list of lowercase tokens.
    """
    if pd.isna(keywords_str):
        return []
    return [kw.strip().lower() for kw in keywords_str.split(";")]

# Create binary columns for each SDOH term.
for term in SDOH_TERMS:
    col_name = f"SDOH_{term.replace(' ', '_')}"
    df[col_name] = df["Keywords"].apply(
        lambda x: 1 if any(term in token for token in tokenize_keywords(x)) else 0
    )

# Create binary columns for each Risk Factor term.
for term in RISK_TERMS:
    col_name = f"RISK_{term.replace(' ', '_')}"
    df[col_name] = df["Keywords"].apply(
        lambda x: 1 if any(term in token for token in tokenize_keywords(x)) else 0
    )

# --- Step 3: Compute the Correlation Matrix ---
# List of newly created binary columns.
sdoh_cols = [f"SDOH_{term.replace(' ', '_')}" for term in SDOH_TERMS]
risk_cols = [f"RISK_{term.replace(' ', '_')}" for term in RISK_TERMS]

# Combine the columns for correlation analysis.
analysis_cols = sdoh_cols + risk_cols

# Compute the Pearson correlation matrix on these binary variables.
corr_matrix = df[analysis_cols].corr(method="pearson")

# --- Step 4: Extract Significant Correlations ---
threshold = 0.05  # set your threshold for significance (|r| > threshold)
sdoh_risk_edges = []

for s_col in sdoh_cols:
    for r_col in risk_cols:
        r_value = corr_matrix.loc[s_col, r_col]
        if abs(r_value) > threshold:
            sdoh_risk_edges.append({
                "source": s_col,
                "target": r_col,
                "relationship": "correlates_with",
                "correlation": r_value,
                "data_source": "CDC_SVI"
            })

# --- Step 5: Output the Results ---
print(f"Significant SDOH-Risk correlations (|r| > {threshold}):")
for edge in sdoh_risk_edges:
    print(edge)
