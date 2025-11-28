# AI Development Workflow Assignment

## Completed Solution for Hospital Re-admission Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Assignment Structure](#assignment-structure)
3. [Installation & Setup](#installation--setup)
4. [Running the Code](#running-the-code)
5. [Repository Structure](#repository-structure)
6. [Key Deliverables](#key-deliverables)
7. [Grading Rubric Alignment](#grading-rubric-alignment)
8. [Technologies Used](#technologies-used)
9. [References](#references)

---

## 🎯 Overview

This repository contains a comprehensive solution to the AI Development Workflow assignment, demonstrating the complete lifecycle of developing, deploying, and maintaining a machine learning system in healthcare.

**Problem Statement:** Develop an AI system to predict hospital adimission risk within 30 days of discharge, enabling pro-active interventions to improve patient outcomes and reduce healthcare costs.

**Key Objectives:**
- Achieve ≥85% recall (sensitivity) for identifying high-risk patients
- Maintain fairness across demographic groups (<5% disparity)
- Ensure HIPAA ( health insurance portability and accountability act) compliance and clinical interpretability
- Deploy an actionable system integrated with hospital workflows

---

## 📚 Assignment Structure

### Part 1: Short Answer Questions (30 points)
**File:** `part1/Answers.md`

Covers:
1. Problem Definition (6 pts)
2. Data Collection & Preprocessing (8 pts)
3. Model Development (8 pts)
4. Evaluation & Deployment (8 pts)

### Part 2: Case Study Application (40 points)
**File:** `part2/CaseStudy_Hospital_Readmission.md`

Comprehensive analysis including:
- Problem scope and stakeholder analysis (5 pts)
- Data strategy with ethical considerations (10 pts)
- Model development with confusion matrix analysis (10 pts)
- Deployment plan with HIPAA compliance (10 pts)
- Overfitting mitigation strategies (5 pts)

### Part 3: Critical Thinking (20 points)
**File:** `part3/CriticalThinking.md`

In-depth discussion of:
- Ethics & Bias: Impact of biased data on patient outcomes (10 pts)
- Trade-offs: Interpretability vs. accuracy, resource constraints (10 pts)

### Part 4: Reflection & Workflow Diagram (10 points)
**File:** `part4/Reflection_Workflow.md`

Contains:
- Reflection on challenges and improvements (5 pts)
- Comprehensive workflow diagram with all stages (5 pts)

### Code Implementation
**Directory:** `code/`

- `hospital_readmission_model.py`: Complete ML pipeline
- `workflow_diagram.py`: Automated diagram generation
- `requirements.txt`: Python dependencies

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/TumainiC/ai_development_workflow.git
cd ai_development_workflow
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r code/requirements.txt
```

**Required Packages:**
- numpy
- pandas
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- shap

---

## ▶️ Running the Code

### 1. Train and Evaluate the Model

```bash
cd code
python hospital_readmission_model.py
```

**Expected Output:**
- Console output with training progress and evaluation metrics
- `results_plots.png`: Confusion matrix, ROC curve, precision-recall curve
- `shap_summary.png`: Feature importance explanations

**Runtime:** ~2-5 minutes depending on your system

### 2. Generate Workflow Diagrams

```bash
python workflow_diagram.py
```

**Expected Output:**
- `ai_workflow_diagram.png`: Detailed workflow with all stages
- `ai_workflow_simple.png`: Simplified flowchart

- TEAM MEMBERS
- Cindy Tumaini
- Hadiza Mohammed

TEAM MEMBERS
CINDY TUMAINI 
HADIZA MOHAMMED

TEAM MEMBERS 
Cindy Tumaini
Hadiza Mohammed
