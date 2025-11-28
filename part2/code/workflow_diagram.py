"""
AI Development Workflow Diagram Generator
Creates visual flowchart of the complete AI development process

Author: AI Development Workflow Assignment
Date: November 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_workflow_diagram():
    """
    Create a comprehensive visual diagram of the AI development workflow.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'problem': '#3498db',      # Blue
        'data': '#2ecc71',          # Green
        'model': '#e74c3c',         # Red
        'eval': '#f39c12',          # Orange
        'deploy': '#9b59b6',        # Purple
        'monitor': '#1abc9c',       # Teal
        'text': '#2c3e50'           # Dark gray
    }
    
    y_pos = 27
    
    # Title
    ax.text(5, y_pos, 'AI DEVELOPMENT WORKFLOW', 
            fontsize=24, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))
    
    y_pos -= 1.5
    ax.text(5, y_pos, 'Hospital Readmission Prediction Case Study', 
            fontsize=14, ha='center', style='italic')
    
    y_pos -= 2
    
    # Stage 1: Problem Definition
    stage_height = 2.5
    box1 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1", 
                          edgecolor=colors['problem'], facecolor=colors['problem'],
                          linewidth=3, alpha=0.3)
    ax.add_patch(box1)
    
    ax.text(1, y_pos - 0.3, 'STAGE 1: PROBLEM DEFINITION', 
            fontsize=12, fontweight='bold', color=colors['problem'])
    ax.text(1, y_pos - 0.8, '• Define scope: Predict 30-day readmission risk', fontsize=9)
    ax.text(1, y_pos - 1.1, '• Stakeholders: Clinicians, patients, administrators', fontsize=9)
    ax.text(1, y_pos - 1.4, '• Objectives: ≥85% recall, <500ms latency', fontsize=9)
    ax.text(1, y_pos - 1.7, '• KPIs: Recall, precision, AUC-ROC, readmission rate', fontsize=9)
    ax.text(1, y_pos - 2.1, '⏱ Timeline: 2 weeks', fontsize=8, style='italic')
    
    # Arrow
    arrow1 = FancyArrowPatch((5, y_pos - stage_height - 0.1), (5, y_pos - stage_height - 0.4),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow1)
    
    y_pos -= stage_height + 0.5
    
    # Stage 2: Data Collection & Preprocessing
    stage_height = 3.5
    box2 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['data'], facecolor=colors['data'],
                          linewidth=3, alpha=0.3)
    ax.add_patch(box2)
    
    ax.text(1, y_pos - 0.3, 'STAGE 2: DATA COLLECTION & PREPROCESSING', 
            fontsize=12, fontweight='bold', color=colors['data'])
    ax.text(1, y_pos - 0.8, 'Data Sources: EHR records, demographics, SDOH data', fontsize=9)
    ax.text(1, y_pos - 1.1, 'Preprocessing Pipeline:', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 1.4, '1. Data integration & deduplication', fontsize=8)
    ax.text(1.5, y_pos - 1.7, '2. Missing data handling (imputation, indicators)', fontsize=8)
    ax.text(1.5, y_pos - 2.0, '3. Feature engineering (comorbidity scores, temporal)', fontsize=8)
    ax.text(1.5, y_pos - 2.3, '4. Encoding & normalization', fontsize=8)
    ax.text(1.5, y_pos - 2.6, '5. Class imbalance handling (SMOTE)', fontsize=8)
    ax.text(1, y_pos - 3.0, 'Bias Mitigation: Stratified sampling, fairness checks', fontsize=9)
    ax.text(1, y_pos - 3.3, '⏱ Timeline: 6-8 weeks', fontsize=8, style='italic')
    
    arrow2 = FancyArrowPatch((5, y_pos - stage_height - 0.1), (5, y_pos - stage_height - 0.4),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow2)
    
    y_pos -= stage_height + 0.5
    
    # Stage 3: Data Splitting
    stage_height = 2.0
    box3 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['data'], facecolor=colors['data'],
                          linewidth=3, alpha=0.2)
    ax.add_patch(box3)
    
    ax.text(1, y_pos - 0.3, 'STAGE 3: DATA SPLITTING', 
            fontsize=12, fontweight='bold', color=colors['data'])
    
    # Three boxes for train/val/test
    ax.text(2, y_pos - 0.9, 'Train\n60%\nN=30K', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue'))
    ax.text(5, y_pos - 0.9, 'Val\n20%\nN=10K', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))
    ax.text(8, y_pos - 0.9, 'Test\n20%\nN=10K', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    ax.text(1, y_pos - 1.6, 'Temporal split • Stratified sampling • Speaker-disjoint', fontsize=8)
    ax.text(1, y_pos - 1.9, '⏱ Timeline: 1 week', fontsize=8, style='italic')
    
    arrow3 = FancyArrowPatch((5, y_pos - stage_height - 0.1), (5, y_pos - stage_height - 0.4),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow3)
    
    y_pos -= stage_height + 0.5
    
    # Stage 4: Model Development
    stage_height = 3.0
    box4 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['model'], facecolor=colors['model'],
                          linewidth=3, alpha=0.3)
    ax.add_patch(box4)
    
    ax.text(1, y_pos - 0.3, 'STAGE 4: MODEL DEVELOPMENT', 
            fontsize=12, fontweight='bold', color=colors['model'])
    ax.text(1, y_pos - 0.8, 'Model: XGBoost (Gradient Boosting Machine)', fontsize=9)
    ax.text(1, y_pos - 1.1, 'Training Process:', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 1.4, '→ Baseline → Hyperparameter tuning → Fairness constraints', fontsize=8)
    ax.text(1.5, y_pos - 1.7, '→ Regularization → Early stopping → Validation', fontsize=8)
    ax.text(1, y_pos - 2.1, 'Key Hyperparameters: learning_rate, max_depth, n_estimators,', fontsize=9)
    ax.text(1, y_pos - 2.4, '                      regularization, scale_pos_weight', fontsize=9)
    ax.text(1, y_pos - 2.8, '⏱ Timeline: 4-6 weeks', fontsize=8, style='italic')
    
    arrow4 = FancyArrowPatch((5, y_pos - stage_height - 0.1), (5, y_pos - stage_height - 0.4),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow4)
    
    y_pos -= stage_height + 0.5
    
    # Stage 5: Model Evaluation
    stage_height = 3.2
    box5 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['eval'], facecolor=colors['eval'],
                          linewidth=3, alpha=0.3)
    ax.add_patch(box5)
    
    ax.text(1, y_pos - 0.3, 'STAGE 5: MODEL EVALUATION', 
            fontsize=12, fontweight='bold', color=colors['eval'])
    ax.text(1, y_pos - 0.8, 'Performance Metrics:', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 1.1, '• Recall: 85% | Precision: 58.6% | F1: 69.3% | AUC: 0.85', fontsize=8)
    ax.text(1, y_pos - 1.5, 'Fairness Evaluation: Performance by demographic groups', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 1.8, '• Max recall disparity: <5% (acceptable)', fontsize=8)
    ax.text(1, y_pos - 2.2, 'Interpretability: SHAP values, feature importance', fontsize=9, fontweight='bold')
    ax.text(1, y_pos - 2.6, '✓ Decision: Model meets success criteria → Deploy', fontsize=9, color='green', fontweight='bold')
    ax.text(1, y_pos - 3.0, '⏱ Timeline: 2-3 weeks', fontsize=8, style='italic')
    
    # Decision diamond
    ax.text(8, y_pos - 1.5, 'Approved?', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='orange', linewidth=2))
    
    arrow5 = FancyArrowPatch((5, y_pos - stage_height - 0.1), (5, y_pos - stage_height - 0.4),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='green')
    ax.add_patch(arrow5)
    
    y_pos -= stage_height + 0.5
    
    # Stage 6: Deployment
    stage_height = 3.5
    box6 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['deploy'], facecolor=colors['deploy'],
                          linewidth=3, alpha=0.3)
    ax.add_patch(box6)
    
    ax.text(1, y_pos - 0.3, 'STAGE 6: DEPLOYMENT', 
            fontsize=12, fontweight='bold', color=colors['deploy'])
    ax.text(1, y_pos - 0.8, 'Phase 1: Pre-Deployment', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 1.1, '• Model containerization (Docker) • API development', fontsize=8)
    ax.text(1.5, y_pos - 1.4, '• EHR integration (HL7/FHIR) • Security audit', fontsize=8)
    ax.text(1, y_pos - 1.8, 'Phase 2: Pilot Deployment', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 2.1, '• Shadow mode → Soft launch → User feedback', fontsize=8)
    ax.text(1, y_pos - 2.5, 'Phase 3: Full Production', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 2.8, '• Hospital-wide rollout • Workflow integration', fontsize=8)
    ax.text(1, y_pos - 3.2, '✓ HIPAA Compliance: Encryption, access controls, audit logs', fontsize=8, color='blue')
    ax.text(1, y_pos - 3.4, '⏱ Timeline: 8-12 weeks', fontsize=8, style='italic')
    
    arrow6 = FancyArrowPatch((5, y_pos - stage_height - 0.1), (5, y_pos - stage_height - 0.4),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow6)
    
    y_pos -= stage_height + 0.5
    
    # Stage 7: Monitoring & Maintenance
    stage_height = 3.0
    box7 = FancyBboxPatch((0.5, y_pos - stage_height), 9, stage_height,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['monitor'], facecolor=colors['monitor'],
                          linewidth=3, alpha=0.3)
    ax.add_patch(box7)
    
    ax.text(1, y_pos - 0.3, 'STAGE 7: MONITORING & MAINTENANCE', 
            fontsize=12, fontweight='bold', color=colors['monitor'])
    ax.text(1, y_pos - 0.8, 'Continuous Monitoring:', fontsize=9, fontweight='bold')
    ax.text(1.5, y_pos - 1.1, '• Performance tracking (weekly/monthly)', fontsize=8)
    ax.text(1.5, y_pos - 1.4, '• Concept drift detection & alerts', fontsize=8)
    ax.text(1.5, y_pos - 1.7, '• Fairness metrics across demographics', fontsize=8)
    ax.text(1, y_pos - 2.1, 'Model Updates: Quarterly retraining • A/B testing • Gradual rollout', fontsize=9)
    ax.text(1, y_pos - 2.5, 'Governance: Ethics review • Bias audits • Adverse event monitoring', fontsize=9)
    ax.text(1, y_pos - 2.9, '⏱ Timeline: Ongoing', fontsize=8, style='italic')
    
    # Feedback loop arrow
    arrow_feedback = FancyArrowPatch((9.3, y_pos - 1.5), (9.3, y_pos + 15),
                                    arrowstyle='->', mutation_scale=20, 
                                    linewidth=2, color='red', linestyle='dashed')
    ax.add_patch(arrow_feedback)
    ax.text(9.5, y_pos + 7, 'Retrain', fontsize=8, rotation=90, va='center', color='red')
    
    # Legend
    legend_y = 1.5
    ax.text(5, legend_y, 'Key Principles', fontsize=11, fontweight='bold', ha='center')
    ax.text(1, legend_y - 0.5, '✓ Iterative Process: Continuous improvement cycles', fontsize=8)
    ax.text(1, legend_y - 0.8, '✓ Human-Centered: Clinical involvement at every stage', fontsize=8)
    ax.text(1, legend_y - 1.1, '✓ Fairness First: Bias mitigation throughout pipeline', fontsize=8)
    ax.text(6, legend_y - 0.5, '✓ Compliance by Design: HIPAA from day one', fontsize=8)
    ax.text(6, legend_y - 0.8, '✓ Continuous Learning: Model adapts over time', fontsize=8)
    ax.text(6, legend_y - 1.1, '✓ Risk Management: Multiple validation gates', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ai_workflow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Workflow diagram saved to ai_workflow_diagram.png")
    plt.close()


def create_simple_flowchart():
    """
    Create a simplified flowchart version for quick reference.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    stages = [
        ("Problem\nDefinition", 14.5, '#3498db'),
        ("Data Collection &\nPreprocessing", 12.5, '#2ecc71'),
        ("Data Splitting", 10.5, '#27ae60'),
        ("Model\nDevelopment", 8.5, '#e74c3c'),
        ("Model\nEvaluation", 6.5, '#f39c12'),
        ("Deployment", 4.5, '#9b59b6'),
        ("Monitoring &\nMaintenance", 2.5, '#1abc9c')
    ]
    
    for i, (stage, y, color) in enumerate(stages):
        # Box
        box = FancyBboxPatch((2, y - 0.7), 6, 1.4,
                            boxstyle="round,pad=0.1",
                            edgecolor=color, facecolor=color,
                            linewidth=2, alpha=0.5)
        ax.add_patch(box)
        
        # Text
        ax.text(5, y, stage, fontsize=12, fontweight='bold', ha='center', va='center')
        
        # Arrow
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((5, y - 0.8), (5, y - 1.6),
                                   arrowstyle='->', mutation_scale=25, 
                                   linewidth=2, color='black')
            ax.add_patch(arrow)
    
    # Title
    ax.text(5, 15.5, 'AI Development Workflow', fontsize=16, fontweight='bold', ha='center')
    
    # Feedback arrow
    arrow_back = FancyArrowPatch((8.5, 3), (8.5, 13.5),
                                arrowstyle='->', mutation_scale=20,
                                linewidth=2, color='red', linestyle='dashed')
    ax.add_patch(arrow_back)
    ax.text(9, 8, 'Iterate', fontsize=10, rotation=90, va='center', color='red')
    
    plt.tight_layout()
    plt.savefig('ai_workflow_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Simple flowchart saved to ai_workflow_simple.png")
    plt.close()


if __name__ == "__main__":
    print("Generating AI Development Workflow Diagrams...")
    print("-" * 50)
    create_workflow_diagram()
    create_simple_flowchart()
    print("\n" + "="*50)
    print("Diagram generation complete!")
    print("Files created:")
    print("  • ai_workflow_diagram.png (detailed)")
    print("  • ai_workflow_simple.png (simplified)")
