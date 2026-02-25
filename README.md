# Feature Scaling — Standardization & Min-Max Scaling

This repository demonstrates **Feature Scaling techniques** used in Machine Learning to normalize numerical data.  
I implemented two separate examples:

- **Standardization** → using `Social_Network_Ads.csv`
- **Min-Max Scaling** → using `Wine_data.csv`

These examples help understand when and why scaling is important before training ML models.

---

## Why Feature Scaling?
Feature scaling ensures that all numerical features contribute equally to model training.  
Without scaling:
- Features with larger values dominate
- Distance-based algorithms perform poorly
- Training becomes slower or unstable

---

# 1. Standardization (Z-Score Scaling)

**File:** `Standardization-Scaling.ipynb`  
**Dataset:** `Social_Network_Ads.csv`

### Formula
\[
Z = \frac{X - \mu}{\sigma}
\]

Where:
- **μ** = mean of feature  
- **σ** = standard deviation  

### Characteristics
- Mean = 0  
- Standard Deviation = 1  
- Handles outliers better than Min-Max scaling

### When to Use
Use Standardization when:
- Data follows normal distribution
- Algorithms depend on distance or gradients

**Examples:**
- Logistic Regression  
- SVM  
- KNN  
- Neural Networks  

---

# 2. Min-Max Scaling (Normalization)

**File:** `Normalization-MinMaxScaler.ipynb`  
**Dataset:** `wine_data.csv`

### Formula
\[
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
\]

### Characteristics
- Values scaled between **0 and 1**
- Preserves original distribution shape
- Sensitive to outliers

### When to Use
Use Min-Max Scaling when:
- Data has fixed boundaries
- You want values in a specific range
- Algorithm does not assume distribution

**Examples:**
- Image processing
- Deep learning inputs
- Gradient descent optimization
