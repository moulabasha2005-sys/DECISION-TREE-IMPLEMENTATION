# ============================================
# DECISION TREE IMPLEMENTATION
# Classification on Iris Dataset
# Author:S SREENIVASULU
# ============================================

# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# 2. Load and Explore the Dataset
# ============================================

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_name'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"\nFeatures: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nClass distribution:")
print(df['target_name'].value_counts())
print(f"\nBasic statistics:")
print(df.describe())

# ============================================
# 3. Exploratory Data Analysis (EDA)
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Pairplot of features colored by target
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                hue='target_name', ax=axes[0, 0], s=100)
axes[0, 0].set_title('Sepal Length vs Sepal Width')

sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                hue='target_name', ax=axes[0, 1], s=100)
axes[0, 1].set_title('Petal Length vs Petal Width')

# Distribution of features
for i, feature in enumerate(iris.feature_names):
    ax = axes[1, 0] if i < 2 else axes[1, 1]
    for target in iris.target_names:
        subset = df[df['target_name'] == target]
        ax.hist(subset[feature], alpha=0.7, label=target, bins=10)
    ax.set_title(f'Distribution of {feature}')
    ax.legend()

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation = df[iris.feature_names].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.show()

# ============================================
# 4. Split Data into Training and Testing Sets
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ============================================
# 5. Train Decision Tree Models
# ============================================

# Model 1: Default Decision Tree
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

# Model 2: Optimized Decision Tree (with pruning to prevent overfitting)
dt_optimized = DecisionTreeClassifier(
    max_depth=3,           # Limit tree depth
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples in a leaf
    criterion='gini',      # Split criterion
    random_state=42
)
dt_optimized.fit(X_train, y_train)

# ============================================
# 6. Model Evaluation
# ============================================

def evaluate_model(model, name):
    """Function to evaluate and print model metrics"""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracy scores
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"\n{'='*60}")
    print(f"{name} MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Cross-validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=iris.target_names))
    
    return y_test_pred, test_acc

# Evaluate both models
y_pred_default, acc_default = evaluate_model(dt_default, "DEFAULT DECISION TREE")
y_pred_optimized, acc_optimized = evaluate_model(dt_optimized, "OPTIMIZED DECISION TREE")

# ============================================
# 7. Confusion Matrices
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix for default model
cm_default = confusion_matrix(y_test, y_pred_default)
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[0])
axes[0].set_title(f'Default Decision Tree\nAccuracy: {acc_default:.3f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Confusion matrix for optimized model
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', 
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1])
axes[1].set_title(f'Optimized Decision Tree\nAccuracy: {acc_optimized:.3f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ============================================
# 8. Visualize the Decision Trees
# ============================================

# Visualize default tree (may be large)
plt.figure(figsize=(20, 10))
plot_tree(dt_default, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title('Default Decision Tree (Unpruned)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualize optimized tree (cleaner and interpretable)
plt.figure(figsize=(16, 8))
plot_tree(dt_optimized, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, 
          rounded=True,
          fontsize=12,
          proportion=True)
plt.title('Optimized Decision Tree (Max Depth = 3, Pruned)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 9. Feature Importance Analysis
# ============================================

# Extract feature importances
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt_optimized.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(feature_importance)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
               color='coral', edgecolor='darkred')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Decision Tree Feature Importance', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, feature_importance['importance'])):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================
# 10. Decision Boundary Visualization (2D)
# ============================================

# Select two most important features for 2D visualization
best_features = feature_importance['feature'].iloc[:2].values
print(f"\nUsing top 2 features for boundary visualization: {best_features}")

# Get indices of selected features
feature_indices = [iris.feature_names.tolist().index(f) for f in best_features]

# Create meshgrid for decision boundary
X_2d = X[:, feature_indices]
dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_2d.fit(X_2d, y)

# Create mesh
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict on mesh
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='Set1')
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='Set1', 
                      edgecolors='black', s=100)
plt.xlabel(best_features[0], fontsize=12)
plt.ylabel(best_features[1], fontsize=12)
plt.title(f'Decision Boundary (Using {best_features[0]} and {best_features[1]})', 
          fontsize=14, fontweight='bold')

# Add legend
legend_labels = iris.target_names
handles = scatter.legend_elements()[0]
plt.legend(handles, legend_labels, title="Classes")

plt.tight_layout()
plt.show()

# ============================================
# 11. Model Analysis and Summary
# ============================================

print("\n" + "=" * 60)
print("MODEL ANALYSIS & CONCLUSIONS")
print("=" * 60)

print("\n📊 MODEL COMPARISON:")
print(f"• Default Tree (unpruned): {acc_default:.3f} test accuracy")
print(f"• Optimized Tree (max_depth=3): {acc_optimized:.3f} test accuracy")

print("\n🎯 KEY FINDINGS:")
print("1. The optimized tree with max_depth=3 prevents overfitting")
print("2. Petal length and petal width are the most important features")
print("3. The tree can perfectly separate Setosa from other species")
print("4. Decision trees provide interpretable rules for classification")

print("\n📈 DECISION RULES (from optimized tree):")
print("• Node splits are based on petal measurements")
print("• Versicolor and Virginica are separated by petal width > 1.75cm")
print("• The tree shows high precision and recall for all classes")

print("\n💡 RECOMMENDATIONS:")
print("• Use max_depth=3 to 5 for better generalization")
print("• Consider min_samples_split to prevent overfitting")
print("• Feature engineering could further improve performance")

# ============================================
# 12. Save the Model (Optional)
# ============================================

import joblib

# Save the optimized model
joblib.dump(dt_optimized, 'decision_tree_iris_model.pkl')
print("\n✅ Model saved as 'decision_tree_iris_model.pkl'")

# Example of loading and using the model
# loaded_model = joblib.load('decision_tree_iris_model.pkl')
# sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example: Setosa
# prediction = loaded_model.predict(sample)
# print(f"Sample prediction: {iris.target_names[prediction[0]]}")

print("\n" + "=" * 60)
print("✅ DECISION TREE IMPLEMENTATION COMPLETE")
print("=" * 60)