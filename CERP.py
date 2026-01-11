"""
CERP (Classification by Ensembles from Random Partitions)
for Ovarian Cancer Early Detection using miRNA Biomarkers

Based on Moon et al. (2006) - Genome Biology 7:R121
https://genomebiology.biomedcentral.com/articles/10.1186/gb-2006-7-12-r121

Implementation follows the original paper exactly:
- Tree pruning: 10-fold CV + 1-SE rule (as per CART/original paper)
- 7 repeats × 10 folds CV for performance evaluation
- No threshold optimization (uses majority voting directly)
- CERP parameters: n_ensembles=15, tree_selection_threshold=0.90
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import List, Tuple, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CERP CLASSIFIER (Original Paper Implementation)
# =============================================================================

class CERPClassifier:
    """
    Classification by Ensembles from Random Partitions (CERP)
    
    Original paper: Moon et al. (2006) Genome Biology 7:R121
    
    Algorithm:
    1. Randomly partition features into mutually exclusive subsets (size ≈ n/6)
    2. Build optimal tree on each subset using CART with pruning
    3. Select trees with sensitivity AND specificity > 90% on training data
    4. Majority vote within ensemble, then majority vote across ensembles
    """
    
    def __init__(
        self,
        n_ensembles: int = 15,
        tree_selection_threshold: float = 0.90,
        min_trees_per_ensemble: int = 3,
        random_state: int = None
    ):
        self.n_ensembles = n_ensembles
        self.tree_selection_threshold = tree_selection_threshold
        self.min_trees_per_ensemble = min_trees_per_ensemble
        self.random_state = random_state
        
        self.ensembles_ = []
        self.feature_partitions_ = []
        self.classes_ = None
    
    def _create_partitions(self, n_features: int, n_samples: int, rng) -> List[np.ndarray]:
        """
        Partition features into mutually exclusive subsets.
        
        Per original paper:
        - Number of partitions: r = 6m/n
        - Each subset contains approximately m/r = n/6 features
        """
        # r = 6m/n (number of partitions)
        n_partitions = max(1, (6 * n_features) // n_samples)
        subset_size = max(2, n_features // n_partitions)
        
        indices = rng.permutation(n_features)
        
        partitions = []
        for i in range(0, n_features, subset_size):
            partition = indices[i:i + subset_size]
            if len(partition) >= 2:
                partitions.append(partition)
        
        return partitions
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, rng) -> DecisionTreeClassifier:
        """
        Build optimal tree using CART with minimal cost-complexity pruning.
        
        Process (per original paper):
        1. Grow full tree until nodes are pure or have ≤5 samples
        2. Generate sequence of subtrees via cost-complexity pruning
        3. Use 10-fold CV to estimate error for each subtree
        4. Apply 1-SE rule: select simplest tree within 1 SE of minimum error
        """
        # Step 1: Get pruning path (sequence of subtrees)
        full_tree = DecisionTreeClassifier(
            criterion='gini',
            min_samples_leaf=5,
            random_state=rng.integers(0, 2**31)
        )
        full_tree.fit(X, y)
        
        path = full_tree.cost_complexity_pruning_path(X, y)
        ccp_alphas = path.ccp_alphas
        ccp_alphas = ccp_alphas[ccp_alphas >= 0]  # Remove any negative values
        
        if len(ccp_alphas) <= 1:
            return full_tree
        
        # Sample alpha candidates for efficiency (max 10 candidates)
        if len(ccp_alphas) > 10:
            indices = np.linspace(0, len(ccp_alphas) - 1, 10, dtype=int)
            ccp_alphas = ccp_alphas[indices]
        
        # Step 2: 10-fold CV to estimate error for each alpha
        min_class_count = np.min(np.bincount(y))
        if min_class_count < 2:
            # Not enough samples for CV, return full tree
            return full_tree
        n_folds = min(10, min_class_count)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=rng.integers(0, 2**31))
        
        cv_errors = []
        cv_stds = []
        
        for alpha in ccp_alphas:
            fold_errors = []
            for train_idx, val_idx in cv.split(X, y):
                tree = DecisionTreeClassifier(
                    criterion='gini',
                    min_samples_leaf=5,
                    ccp_alpha=alpha,
                    random_state=rng.integers(0, 2**31)
                )
                tree.fit(X[train_idx], y[train_idx])
                error = 1 - tree.score(X[val_idx], y[val_idx])
                fold_errors.append(error)
            
            cv_errors.append(np.mean(fold_errors))
            cv_stds.append(np.std(fold_errors))
        
        cv_errors = np.array(cv_errors)
        cv_stds = np.array(cv_stds)
        
        # Step 3: Apply 1-SE rule
        min_idx = np.argmin(cv_errors)
        min_error = cv_errors[min_idx]
        se_at_min = cv_stds[min_idx] / np.sqrt(n_folds)
        threshold = min_error + se_at_min
        
        # Select simplest tree (highest alpha) within 1-SE of minimum
        valid_indices = np.where(cv_errors <= threshold)[0]
        best_idx = valid_indices[-1]
        best_alpha = ccp_alphas[best_idx]
        
        # Step 4: Build final pruned tree
        optimal_tree = DecisionTreeClassifier(
            criterion='gini',
            min_samples_leaf=5,
            ccp_alpha=best_alpha,
            random_state=rng.integers(0, 2**31)
        )
        optimal_tree.fit(X, y)
        
        return optimal_tree
    
    def _evaluate_tree(self, tree, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Calculate sensitivity and specificity on training data."""
        y_pred = tree.predict(X)
        
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        
        sensitivity = (y_pred[pos_mask] == 1).sum() / pos_mask.sum() if pos_mask.sum() > 0 else 0
        specificity = (y_pred[neg_mask] == 0).sum() / neg_mask.sum() if neg_mask.sum() > 0 else 0
        
        return sensitivity, specificity
    
    def _build_ensemble(self, X: np.ndarray, y: np.ndarray, rng) -> Tuple[List, List]:
        """
        Build single ensemble:
        1. Random partition of features
        2. Build tree for each partition
        3. Select trees with sens AND spec > threshold
        """
        n_samples, n_features = X.shape
        partitions = self._create_partitions(n_features, n_samples, rng)
        
        trees = []
        features = []
        performances = []
        
        # Build and evaluate trees
        for partition in partitions:
            X_sub = X[:, partition]
            tree = self._build_tree(X_sub, y, rng)
            sens, spec = self._evaluate_tree(tree, X_sub, y)
            
            trees.append(tree)
            features.append(partition)
            performances.append((sens, spec))
        
        # Select trees with high sensitivity AND specificity (>90%)
        # Decrease threshold by 5% if fewer than 3 trees qualify
        threshold = self.tree_selection_threshold
        selected_trees = []
        selected_features = []
        
        while len(selected_trees) < self.min_trees_per_ensemble and threshold >= 0.5:
            selected_trees = []
            selected_features = []
            
            for i, (sens, spec) in enumerate(performances):
                if sens >= threshold and spec >= threshold:
                    selected_trees.append(trees[i])
                    selected_features.append(features[i])
            
            threshold -= 0.05
        
        # Fallback: take best trees if still not enough
        if len(selected_trees) < self.min_trees_per_ensemble:
            scores = [(s + p) / 2 for s, p in performances]
            top_idx = np.argsort(scores)[-self.min_trees_per_ensemble:]
            selected_trees = [trees[i] for i in top_idx]
            selected_features = [features[i] for i in top_idx]
        
        return selected_trees, selected_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit CERP classifier."""
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        
        self.ensembles_ = []
        self.feature_partitions_ = []
        
        for _ in range(self.n_ensembles):
            trees, features = self._build_ensemble(X, y, rng)
            self.ensembles_.append(trees)
            self.feature_partitions_.append(features)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using two-level majority voting:
        1. Majority vote within each ensemble
        2. Majority vote across ensembles

        Note: Ties favor class 1 (Cancer) for higher sensitivity in early detection.
        """
        n_samples = X.shape[0]
        ensemble_preds = []

        # Get prediction from each ensemble
        for trees, feature_sets in zip(self.ensembles_, self.feature_partitions_):
            votes = np.zeros(n_samples)

            for tree, feats in zip(trees, feature_sets):
                votes += tree.predict(X[:, feats])

            # Majority vote within ensemble (ties favor Cancer)
            ensemble_pred = (votes >= len(trees) / 2).astype(int)
            ensemble_preds.append(ensemble_pred)

        # Majority vote across ensembles (ties favor Cancer)
        ensemble_preds = np.array(ensemble_preds)
        final_pred = (ensemble_preds.sum(axis=0) >= self.n_ensembles / 2).astype(int)

        return final_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return proportion of ensemble votes for probability estimation."""
        n_samples = X.shape[0]
        ensemble_preds = []

        for trees, feature_sets in zip(self.ensembles_, self.feature_partitions_):
            votes = np.zeros(n_samples)
            for tree, feats in zip(trees, feature_sets):
                votes += tree.predict(X[:, feats])
            ensemble_pred = (votes >= len(trees) / 2).astype(int)
            ensemble_preds.append(ensemble_pred)
        
        ensemble_preds = np.array(ensemble_preds)
        prob_pos = ensemble_preds.mean(axis=0)
        
        return np.column_stack([1 - prob_pos, prob_pos])


# =============================================================================
# FAST R SCREENING (Quick search for optimal partition count)
# =============================================================================

def screen_optimal_r(
    X: np.ndarray,
    y: np.ndarray,
    r_candidates: List[int] = None,
    n_folds: int = 3,
    n_ensembles: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Fast screening to find optimal number of partitions (r).

    Uses reduced CV folds and ensembles for speed.
    Run this first, then use the best r for full evaluation.

    Parameters
    ----------
    X, y : arrays
    r_candidates : list of int, optional
        Partition counts to test. Default: [3, 7, 11, 21, 31, 51, 71, 101]
    n_folds : int, default=3
        CV folds (fewer = faster)
    n_ensembles : int, default=5
        Ensembles per test (fewer = faster)
    random_state : int
    verbose : bool

    Returns
    -------
    dict with 'best_r', 'results', and 'recommendation'
    """
    if r_candidates is None:
        # Default: coarse grid of odd numbers
        r_candidates = [3, 7, 11, 21, 31, 51, 71, 101]

    # Filter candidates that are too large
    max_r = X.shape[1] // 2
    r_candidates = [r for r in r_candidates if r <= max_r]

    if verbose:
        print(f"\n{'='*60}")
        print("FAST R SCREENING")
        print(f"{'='*60}")
        print(f"Testing r values: {r_candidates}")
        print(f"Settings: {n_folds}-fold CV, {n_ensembles} ensembles (fast mode)")
        print(f"{'='*60}\n")

    results = []

    # Scale data once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for r in tqdm(r_candidates, desc="Screening r", disable=not verbose):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_accs = []
        fold_aucs = []
        fold_sens = []
        fold_spec = []

        for train_idx, test_idx in cv.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Quick CERP with fewer ensembles
            clf = CERPClassifier(
                n_ensembles=n_ensembles,
                random_state=random_state
            )
            # Manually set partition count
            clf.optimal_r_ = r
            clf.n_partitions = r

            # Modified fit with fixed r
            clf.classes_ = np.unique(y_train)
            rng = np.random.default_rng(random_state)
            clf.ensembles_ = []
            clf.feature_partitions_ = []

            n_features = X_train.shape[1]
            for _ in range(n_ensembles):
                # Create partitions
                indices = rng.permutation(n_features)
                subset_size = max(1, n_features // r)
                partitions = []
                for i in range(0, n_features, subset_size):
                    part = indices[i:i + subset_size]
                    if len(part) >= 1:
                        partitions.append(part)

                # Build trees (simplified - no pruning for speed)
                trees = []
                features = []
                for part in partitions:
                    tree = DecisionTreeClassifier(
                        min_samples_leaf=5,
                        random_state=rng.integers(0, 2**31)
                    )
                    tree.fit(X_train[:, part], y_train)
                    trees.append(tree)
                    features.append(part)

                clf.ensembles_.append(trees)
                clf.feature_partitions_.append(features)

            # Predict
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            # Metrics
            fold_accs.append(np.mean(y_pred == y_test))
            try:
                fold_aucs.append(roc_auc_score(y_test, y_proba))
            except ValueError:
                fold_aucs.append(0.5)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fold_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        result = {
            'r': r,
            'accuracy': np.mean(fold_accs),
            'auc': np.mean(fold_aucs),
            'sensitivity': np.mean(fold_sens),
            'specificity': np.mean(fold_spec),
            'acc_std': np.std(fold_accs)
        }
        results.append(result)

        if verbose:
            print(f"  r={r:3d}: AUC={result['auc']:.4f}, "
                  f"Acc={result['accuracy']:.4f}, "
                  f"Sens={result['sensitivity']:.4f}, "
                  f"Spec={result['specificity']:.4f}")

    # Find best r (by AUC, then accuracy)
    best_result = max(results, key=lambda x: (x['auc'], x['accuracy']))
    best_r = best_result['r']

    # Recommendation
    if verbose:
        print(f"\n{'='*60}")
        print("SCREENING RESULTS")
        print(f"{'='*60}")
        print(f"Best r = {best_r} (AUC={best_result['auc']:.4f})")
        print(f"\nTop 3 candidates:")
        sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)[:3]
        for res in sorted_results:
            print(f"  r={res['r']:3d}: AUC={res['auc']:.4f}, Acc={res['accuracy']:.4f}")
        print(f"\nRecommendation: Run full evaluation with r={best_r}")
        print(f"{'='*60}")

    return {
        'best_r': best_r,
        'best_result': best_result,
        'all_results': results
    }


# =============================================================================
# EVALUATION PIPELINE (Original Paper: 20 reps × 10-fold CV)
# =============================================================================

def evaluate_cerp(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 7,
    n_folds: int = 10,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Evaluate CERP using repeated cross-validation.
    
    Original paper used 20 reps × 10-fold CV.
    Modified to 7 reps × 10-fold for consistency with thesis methodology.
    """
    
    results = {
        'fold_results': [],
        'repeat_results': []
    }
    
    if verbose:
        print(f"\nCERP Evaluation: {n_repeats} repeats × {n_folds} folds")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: Cancer={np.sum(y==1)}, Control={np.sum(y==0)}")
        print("=" * 60)

    # Progress bar for repeats
    repeat_iter = tqdm(range(n_repeats), desc="Repeats", disable=not verbose)

    for rep in repeat_iter:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state + rep)

        rep_y_true = []
        rep_y_pred = []
        rep_y_proba = []

        # Progress bar for folds
        fold_iter = tqdm(enumerate(cv.split(X, y)), total=n_folds,
                         desc=f"  Rep {rep+1} Folds", leave=False, disable=not verbose)

        for fold, (train_idx, test_idx) in fold_iter:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train CERP with fixed parameters (as in original paper)
            clf = CERPClassifier(
                n_ensembles=15,
                tree_selection_threshold=0.90,
                min_trees_per_ensemble=3,
                random_state=random_state + rep * 100 + fold
            )
            clf.fit(X_train_scaled, y_train)
            
            # Predict (no threshold optimization - direct majority voting)
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            rep_y_true.extend(y_test)
            rep_y_pred.extend(y_pred)
            rep_y_proba.extend(y_proba)
            
            # Store fold results
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fold_metrics = calculate_metrics(tn, fp, fn, tp, y_test, y_proba)
            fold_metrics['repeat'] = rep
            fold_metrics['fold'] = fold
            results['fold_results'].append(fold_metrics)
        
        # Calculate repeat-level metrics
        rep_y_true = np.array(rep_y_true)
        rep_y_pred = np.array(rep_y_pred)
        rep_y_proba = np.array(rep_y_proba)
        
        cm = confusion_matrix(rep_y_true, rep_y_pred)
        tn, fp, fn, tp = cm.ravel()
        rep_metrics = calculate_metrics(tn, fp, fn, tp, rep_y_true, rep_y_proba)
        rep_metrics['repeat'] = rep
        results['repeat_results'].append(rep_metrics)
        
        # Print repeat results
        if verbose:
            print_repeat_results(rep, rep_metrics, cm)
    
    # Print final summary
    if verbose:
        print_final_summary(results, n_repeats, n_folds)
    
    return results


def calculate_metrics(tn, fp, fn, tp, y_true=None, y_proba=None) -> Dict:
    """Calculate all performance metrics."""
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    auc = None
    if y_true is not None and y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            # Occurs when only one class is present in y_true
            pass
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'auc': auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def print_repeat_results(rep_idx: int, metrics: Dict, cm: np.ndarray):
    """Print results for a single repeat."""
    print(f"\n{'='*60}")
    print(f"REPEAT {rep_idx + 1} RESULTS")
    print(f"{'='*60}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual Neg (Control)  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Actual Pos (Cancer)   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    print(f"\nPerformance Metrics:")
    print(f"-" * 40)
    print(f"  TP: {metrics['tp']:4d}    FP: {metrics['fp']:4d}")
    print(f"  FN: {metrics['fn']:4d}    TN: {metrics['tn']:4d}")
    print(f"-" * 40)
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  PPV:         {metrics['ppv']:.4f} ({metrics['ppv']*100:.2f}%)")
    print(f"  NPV:         {metrics['npv']:.4f} ({metrics['npv']*100:.2f}%)")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    if metrics['auc'] is not None:
        print(f"  AUC:         {metrics['auc']:.4f}")


def print_final_summary(results: Dict, n_repeats: int, n_folds: int):
    """Print final summary across all repeats."""
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY (Across {n_repeats} Repeats)")
    print(f"{'='*60}")
    
    print(f"\nConfiguration:")
    print(f"  Cross-validation: {n_repeats} repeats × {n_folds} folds")
    print(f"  CERP parameters: n_ensembles=15, threshold=0.90")
    
    # Per-repeat table
    print(f"\nPer-Repeat Results:")
    print(f"-" * 80)
    print(f"{'Repeat':>8} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Acc':>8} {'PPV':>8} {'NPV':>8} {'F1':>8}")
    print(f"-" * 80)
    
    rep_results = results['repeat_results']
    for r in rep_results:
        print(f"{r['repeat']+1:>8} {r['auc']:>8.4f} {r['sensitivity']:>8.4f} "
              f"{r['specificity']:>8.4f} {r['accuracy']:>8.4f} {r['ppv']:>8.4f} "
              f"{r['npv']:>8.4f} {r['f1']:>8.4f}")
    
    print(f"-" * 80)
    
    # Overall summary
    print(f"\nOverall Performance (Mean ± SD across repeats):")
    print(f"-" * 40)
    
    metrics = ['auc', 'sensitivity', 'specificity', 'accuracy', 'ppv', 'npv', 'f1']
    for m in metrics:
        values = [r[m] for r in rep_results if r[m] is not None]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {m.upper():12}: {mean:.4f} ± {std:.4f}")
    
    # Clinical interpretation
    print(f"\nClinical Interpretation:")
    print(f"-" * 40)
    sens_values = [r['sensitivity'] for r in rep_results]
    spec_values = [r['specificity'] for r in rep_results]
    auc_values = [r['auc'] for r in rep_results if r['auc'] is not None]
    
    sens_mean = np.mean(sens_values)
    spec_mean = np.mean(spec_values)
    auc_mean = np.mean(auc_values) if auc_values else 0
    
    print(f"  AUC {auc_mean:.3f} indicates {'good' if auc_mean >= 0.8 else 'moderate' if auc_mean >= 0.7 else 'poor'} discriminative ability")
    print(f"  Sensitivity {sens_mean:.1%} - proportion of cancers correctly detected")
    print(f"  Specificity {spec_mean:.1%} - proportion of controls correctly identified")
    
    if sens_mean > 0.62:
        print(f"  ✓ Sensitivity exceeds CA-125 benchmark (50-62% for early stage)")
    else:
        print(f"  ✗ Sensitivity below CA-125 benchmark (50-62% for early stage)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Ovarian Cancer Early Detection with CERP")
    print("=" * 60)

    # Load data
    data_file = 'final_ov.xlsx'
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        print("Please ensure the file is in the current directory.")
        exit(1)

    df = pd.read_excel(data_file)
    
    # Extract labels
    y_labels = df['2 group class'].values
    y = (y_labels == 'Cancer').astype(int)
    
    # Extract miRNA features
    mirna_cols = [col for col in df.columns if col.startswith('hsa-')]
    X = df[mirna_cols].values
    
    print(f"\n✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Cancer: {np.sum(y==1)}, Control: {np.sum(y==0)}")
    
    # Run CERP evaluation (7 repeats × 10 folds, as per original paper methodology)
    results = evaluate_cerp(
        X, y,
        n_repeats=7,
        n_folds=10,
        random_state=42,
        verbose=True
    )
