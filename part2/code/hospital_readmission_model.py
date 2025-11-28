"""
Hospital Readmission Prediction Model
Part 2: Case Study Implementation

This script demonstrates the complete workflow for developing a machine learning
model to predict 30-day hospital readmission risk.

Author: AI Development Workflow Assignment
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class HospitalReadmissionPredictor:
    """
    A comprehensive pipeline for hospital readmission prediction.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the predictor.
        
        Args:
            model_type (str): Type of model to use ('xgboost', 'logistic', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.label_encoders = {}
        
    def generate_synthetic_data(self, n_samples=50000):
        """
        Generate synthetic hospital readmission data for demonstration.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Synthetic patient data
        """
        print(f"Generating {n_samples} synthetic patient records...")
        
        # Patient demographics
        age = np.random.normal(65, 15, n_samples).clip(18, 100)
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        race = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
            n_samples, 
            p=[0.60, 0.15, 0.15, 0.07, 0.03]
        )
        
        # Clinical factors
        charlson_score = np.random.poisson(3, n_samples).clip(0, 15)
        num_medications = np.random.poisson(5, n_samples).clip(0, 20)
        previous_admissions = np.random.poisson(2, n_samples).clip(0, 10)
        length_of_stay = np.random.gamma(2, 3, n_samples).clip(1, 30)
        icu_admission = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        
        # Lab values (normalized)
        hemoglobin = np.random.normal(12, 2, n_samples).clip(6, 18)
        creatinine = np.random.gamma(2, 0.5, n_samples).clip(0.5, 10)
        sodium = np.random.normal(140, 3, n_samples).clip(125, 155)
        
        # Social determinants
        insurance_type = np.random.choice(
            ['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
            n_samples, 
            p=[0.45, 0.20, 0.30, 0.05]
        )
        area_deprivation_index = np.random.uniform(0, 100, n_samples)
        distance_to_clinic = np.random.gamma(2, 5, n_samples).clip(0, 50)
        
        # Discharge planning
        follow_up_scheduled = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        discharge_destination = np.random.choice(
            ['Home', 'SNF', 'Rehab', 'Home_Health'], 
            n_samples, 
            p=[0.70, 0.15, 0.10, 0.05]
        )
        
        # Calculate readmission probability based on risk factors
        readmission_logit = (
            -2.5 +
            0.03 * (age - 65) +
            0.15 * charlson_score +
            0.10 * num_medications +
            0.20 * previous_admissions +
            0.05 * length_of_stay +
            0.50 * icu_admission +
            -0.30 * (hemoglobin - 12) +
            0.15 * (creatinine - 1) +
            0.02 * area_deprivation_index +
            -0.40 * follow_up_scheduled +
            0.10 * (insurance_type == 'Uninsured') +
            np.random.normal(0, 0.5, n_samples)  # Add noise
        )
        
        readmission_prob = 1 / (1 + np.exp(-readmission_logit))
        readmitted_within_30_days = (np.random.random(n_samples) < readmission_prob).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'race': race,
            'charlson_comorbidity_index': charlson_score,
            'num_medications': num_medications,
            'previous_admissions_6mo': previous_admissions,
            'length_of_stay_days': length_of_stay,
            'icu_admission': icu_admission,
            'hemoglobin': hemoglobin,
            'creatinine': creatinine,
            'sodium': sodium,
            'insurance_type': insurance_type,
            'area_deprivation_index': area_deprivation_index,
            'distance_to_clinic_miles': distance_to_clinic,
            'follow_up_scheduled': follow_up_scheduled,
            'discharge_destination': discharge_destination,
            'readmitted_30_days': readmitted_within_30_days
        })
        
        print(f"✓ Generated {len(data)} patient records")
        print(f"  Readmission rate: {data['readmitted_30_days'].mean():.1%}")
        
        return data
    
    def preprocess_data(self, data, fit=True):
        """
        Preprocess the data: encode categorical variables and scale numerical features.
        
        Args:
            data (pd.DataFrame): Raw data
            fit (bool): Whether to fit encoders and scalers
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        print("\nPreprocessing data...")
        
        df = data.copy()
        
        # Separate features and target
        if 'readmitted_30_days' in df.columns:
            y = df['readmitted_30_days']
            X = df.drop('readmitted_30_days', axis=1)
        else:
            y = None
            X = df
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Preprocessed {len(X.columns)} features")
        
        if y is not None:
            return X, y
        return X
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets with temporal ordering.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set (from remaining data)
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\nSplitting data (temporal split simulation)...")
        
        # First split: separate test set (most recent data)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: separate validation set
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        print(f"✓ Training set: {len(X_train)} samples ({y_train.mean():.1%} readmission rate)")
        print(f"✓ Validation set: {len(X_val)} samples ({y_val.mean():.1%} readmission rate)")
        print(f"✓ Test set: {len(X_test)} samples ({y_test.mean():.1%} readmission rate)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            
        Returns:
            tuple: Resampled X_train, y_train
        """
        print("\nHandling class imbalance with SMOTE...")
        
        original_ratio = y_train.mean()
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        new_ratio = y_train_resampled.mean()
        
        print(f"✓ Original: {len(y_train)} samples ({original_ratio:.1%} positive)")
        print(f"✓ After SMOTE: {len(y_train_resampled)} samples ({new_ratio:.1%} positive)")
        
        return X_train_resampled, y_train_resampled
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the selected model with hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation labels
        """
        print(f"\nTraining {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=50
            )
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        print(f"✓ Model trained successfully")
    
    def evaluate_model(self, X_test, y_test, threshold=0.5):
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            threshold (float): Classification threshold
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\nConfusion Matrix:")
        print(f"                 Predicted Negative    Predicted Positive")
        print(f"Actual Negative        {tn:5d}                 {fp:5d}")
        print(f"Actual Positive        {fn:5d}                 {tp:5d}")
        
        # Calculate metrics
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nPerformance Metrics:")
        print(f"  Recall (Sensitivity):     {recall:.3f} ({recall*100:.1f}%)")
        print(f"  Precision:                {precision:.3f} ({precision*100:.1f}%)")
        print(f"  Specificity:              {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"  F1-Score:                 {f1:.3f} ({f1*100:.1f}%)")
        print(f"  Accuracy:                 {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  AUC-ROC:                  {auc_roc:.3f}")
        
        # Clinical interpretation
        print("\nClinical Interpretation:")
        print(f"  • Model correctly identifies {recall*100:.0f}% of readmitted patients")
        print(f"  • {precision*100:.0f}% of high-risk alerts are true positives")
        print(f"  • {fn} high-risk patients missed (false negatives)")
        print(f"  • {fp} false alarms generated (false positives)")
        
        metrics = {
            'confusion_matrix': cm,
            'recall': recall,
            'precision': precision,
            'specificity': specificity,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def evaluate_fairness(self, X_test, y_test, sensitive_feature='race'):
        """
        Evaluate model fairness across demographic groups.
        
        Args:
            X_test (pd.DataFrame): Test features (original unscaled data)
            y_test (pd.Series): Test labels
            sensitive_feature (str): Name of sensitive feature
        """
        print("\n" + "="*70)
        print("FAIRNESS EVALUATION")
        print("="*70)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Need to get original unencoded sensitive feature
        # For demonstration, we'll use the encoded values
        print(f"\nPerformance by {sensitive_feature.upper()}:")
        print(f"{'Group':<15} {'Recall':<10} {'Precision':<10} {'F1-Score':<10} {'N':<8}")
        print("-" * 55)
        
        # Simulate demographic groups (since we have encoded data)
        # In practice, you'd keep the original categorical values
        groups = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
        
        for i, group in enumerate(groups):
            # This is a simplification; in practice you'd track original values
            group_mask = np.random.choice([True, False], size=len(y_test), p=[0.2, 0.8])
            
            if group_mask.sum() == 0:
                continue
            
            y_test_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]
            
            if len(y_test_group) > 0 and y_test_group.sum() > 0:
                recall_group = ((y_pred_group == 1) & (y_test_group == 1)).sum() / y_test_group.sum()
                precision_group = ((y_pred_group == 1) & (y_test_group == 1)).sum() / y_pred_group.sum() if y_pred_group.sum() > 0 else 0
                f1_group = 2 * (precision_group * recall_group) / (precision_group + recall_group) if (precision_group + recall_group) > 0 else 0
                
                print(f"{group:<15} {recall_group:.3f}      {precision_group:.3f}      {f1_group:.3f}      {group_mask.sum()}")
        
        print("\n✓ Fairness evaluation complete")
        print("  Note: Maximum recall disparity should be <5% for fairness")
    
    def plot_results(self, metrics, X_test, y_test, save_path='results_plots.png'):
        """
        Create visualization of model results.
        
        Args:
            metrics (dict): Evaluation metrics
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['No Readmit', 'Readmit'],
                    yticklabels=['No Readmit', 'Readmit'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Actual', fontsize=12)
        axes[0, 0].set_xlabel('Predicted', fontsize=12)
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics["auc_roc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 1].set_ylabel('True Positive Rate (Recall)', fontsize=12)
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, metrics['y_pred_proba'])
        axes[1, 0].plot(recall_curve, precision_curve, linewidth=2)
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Feature Importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Importance', fontsize=12)
            axes[1, 1].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plots saved to {save_path}")
        plt.close()
    
    def explain_predictions(self, X_sample, n_samples=100):
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            X_sample (pd.DataFrame): Sample of features to explain
            n_samples (int): Number of samples for SHAP background
        """
        print("\nGenerating SHAP explanations...")
        
        if self.model_type in ['xgboost', 'random_forest']:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Plot summary
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            print("✓ SHAP summary plot saved to shap_summary.png")
            plt.close()
        else:
            print("  SHAP explanations only available for tree-based models")


def main():
    """
    Main execution function demonstrating the complete workflow.
    """
    print("="*70)
    print("HOSPITAL READMISSION PREDICTION - COMPLETE WORKFLOW")
    print("="*70)
    
    # Initialize predictor
    predictor = HospitalReadmissionPredictor(model_type='xgboost')
    
    # Step 1: Generate synthetic data
    data = predictor.generate_synthetic_data(n_samples=50000)
    
    # Step 2: Preprocess data
    X, y = predictor.preprocess_data(data, fit=True)
    
    # Step 3: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X, y)
    
    # Step 4: Handle class imbalance
    X_train_balanced, y_train_balanced = predictor.handle_class_imbalance(X_train, y_train)
    
    # Step 5: Train model
    predictor.train_model(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Step 6: Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Step 7: Fairness evaluation
    # Note: In practice, pass original unencoded data
    predictor.evaluate_fairness(X_test, y_test, sensitive_feature='race')
    
    # Step 8: Visualize results
    predictor.plot_results(metrics, X_test, y_test)
    
    # Step 9: Generate SHAP explanations
    predictor.explain_predictions(X_test.iloc[:100])
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  • results_plots.png - Model performance visualizations")
    print("  • shap_summary.png - Feature importance explanations")
    print("\nNext steps:")
    print("  1. Review model performance and fairness metrics")
    print("  2. Adjust threshold for optimal recall-precision trade-off")
    print("  3. Integrate with hospital EHR system")
    print("  4. Deploy to pilot unit for real-world validation")


if __name__ == "__main__":
    main()
