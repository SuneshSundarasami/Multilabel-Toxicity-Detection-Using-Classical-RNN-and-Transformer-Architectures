import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score

class ModelEvaluator:
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize the model evaluator
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    def evaluate_model(self,
                    model: Any,
                    X_test: Any,
                    y_test: np.ndarray,
                    target_columns: List[str],
                    threshold: float = 0.5,
                    output_name: Optional[str] = None) -> Dict[str, Any]:
        
        # Get raw predictions instead of using predict_proba
        y_pred_raw = model.predict(X_test)
        
        # Calculate metrics
        results = {}
        
        # Calculate accuracy per sample
        accuracy = np.mean(np.all(y_pred_raw == y_test, axis=1))
        results['accuracy'] = accuracy
        
        # Calculate F1 scores for each label individually
        f1_scores = []
        for i in range(len(target_columns)):
            f1 = f1_score(y_test[:, i], y_pred_raw[:, i])
            f1_scores.append(f1)
            results[f'f1_{target_columns[i]}'] = f1
        
        # Calculate macro, micro, and weighted F1 scores
        results['macro_f1'] = np.mean(f1_scores)
        results['micro_f1'] = f1_score(y_test.ravel(), y_pred_raw.ravel())
        results['weighted_f1'] = f1_score(y_test, y_pred_raw, average='weighted')
        
        # For AUC, we need probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            auc_scores = []
            for i in range(len(target_columns)):
                # Extract the probability of the positive class
                if isinstance(y_pred_proba, list):
                    pos_proba = y_pred_proba[i][:, 1]
                else:
                    # Handle case where probabilities might be structured differently
                    pos_proba = y_pred_proba[:, i]
                    
                auc = roc_auc_score(y_test[:, i], pos_proba)
                auc_scores.append(auc)
                results[f'auc_{target_columns[i]}'] = auc
            
            results['mean_auc'] = np.mean(auc_scores)
        else:
            results['mean_auc'] = np.nan
        
        # Print summary
        print(f"Model Evaluation Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")
        if 'mean_auc' in results:
            print(f"Mean AUC: {results['mean_auc']:.4f}")
        
        # Generate classification report for each class
        for i, col in enumerate(target_columns):
            print(f"\nClassification Report for {col}:")
            report = classification_report(y_test[:, i], y_pred_raw[:, i])
            print(report)
            results[f'report_{col}'] = report
        
        # Save detailed results if output_name is provided
        if output_name:
            # Create the results directory if it doesn't exist
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
                
            results_path = os.path.join(self.results_dir, f"{output_name}_results.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Generate and save visualization
            self.plot_results(results, target_columns, output_name)
            
            print(f"Evaluation results saved to {results_path}")
        
        return results

    def plot_results(self, 
                    results: Dict[str, Any], 
                    target_columns: List[str],
                    output_name: str):
        """
        Plot evaluation results
        """
        # Check which metrics are available in the results
        available_metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'mean_auc']
        metrics_to_plot = [m for m in available_metrics if m in results]
        
        # Plot AUC scores by class if available
        if any(f'auc_{col}' in results for col in target_columns):
            plt.figure(figsize=(12, 6))
            auc_values = []
            for col in target_columns:
                if f'auc_{col}' in results:
                    auc_values.append(results[f'auc_{col}'])
                else:
                    auc_values.append(0)  # Default value if not available
            
            bars = plt.bar(target_columns, auc_values, color='skyblue')
            if 'mean_auc' in results:
                plt.axhline(results['mean_auc'], color='red', linestyle='--', 
                        label=f'Mean AUC: {results["mean_auc"]:.4f}')
            
            plt.title('ROC AUC by Class')
            plt.xlabel('Class')
            plt.ylabel('AUC Score')
            plt.ylim(0, 1.05)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.results_dir, f"{output_name}_auc_by_class.png"))
            plt.close()
        
        # Plot overall metrics if any are available
        if metrics_to_plot:
            plt.figure(figsize=(10, 6))
            metric_values = [results[metric] for metric in metrics_to_plot]
            
            bars = plt.bar(metrics_to_plot, metric_values, color='lightgreen')
            
            plt.title('Overall Model Performance Metrics')
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.ylim(0, 1.05)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{output_name}_overall_metrics.png"))
            plt.close()
    
    def compare_models(self,
                      result_files: List[str],
                      model_names: List[str],
                      output_name: str = 'model_comparison'):
        """
        Compare multiple trained models
        
        Args:
            result_files: List of paths to result pickle files
            model_names: List of model names for the legend
            output_name: Name for output comparison file
        """
        if len(result_files) != len(model_names):
            raise ValueError("Number of result files must match number of model names")
        
        # Load results from files
        all_results = []
        for file_path in result_files:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
                all_results.append(results)
        
        # Extract common metrics for comparison
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'mean_auc']
        metrics_data = {}
        
        for metric in metrics:
            metrics_data[metric] = [results[metric] for results in all_results]
        
        # Create comparison plot
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            offset = (i - len(model_names)/2 + 0.5) * width
            values = [all_results[i][metric] for metric in metrics]
            bars = plt.bar(x + offset, values, width, label=model_name)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend(loc='best')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, f"{output_name}.png"))
        plt.close()
        
        print(f"Model comparison plot saved to {os.path.join(self.results_dir, output_name)}.png")
        
        # Create a DataFrame with comparison data
        comparison_df = pd.DataFrame({
            'Model': model_names * len(metrics),
            'Metric': sum([[metric] * len(model_names) for metric in metrics], []),
            'Score': sum([metrics_data[metric] for metric in metrics], [])
        })
        
        # Save comparison data
        comparison_df.to_csv(os.path.join(self.results_dir, f"{output_name}.csv"), index=False)
        
        return comparison_df
    
    def find_optimal_threshold(self,
                              model: Any,
                              X_val: Any,
                              y_val: np.ndarray,
                              target_columns: List[str],
                              thresholds: List[float] = None,
                              metric: str = 'f1',
                              output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Find optimal classification threshold for each class
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
        else:
            print("Model doesn't support probability predictions, using default threshold.")
            return {
                'thresholds': {col: 0.5 for col in target_columns},
                'scores': {col: 0.0 for col in target_columns},
                'mean_threshold': 0.5
            }
        
        optimal_thresholds = {}
        optimal_scores = {}
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 8))
        
        # For each class, find the optimal threshold
        for i, col in enumerate(target_columns):
            scores = []
            
            # Test each threshold
            for threshold in thresholds:
                # Extract the probability for the positive class
                if isinstance(y_pred_proba, list):
                    pos_proba = y_pred_proba[i][:, 1]
                else:
                    # Handle case where probabilities might be structured differently
                    pos_proba = y_pred_proba[:, i]
                    
                y_pred_i = (pos_proba >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_val[:, i], y_pred_i)
                elif metric == 'accuracy':
                    score = accuracy_score(y_val[:, i], y_pred_i)
                elif metric == 'precision':
                    from sklearn.metrics import precision_score
                    score = precision_score(y_val[:, i], y_pred_i)
                elif metric == 'recall':
                    from sklearn.metrics import recall_score
                    score = recall_score(y_val[:, i], y_pred_i)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                scores.append(score)
            
            # Find the optimal threshold
            best_idx = np.argmax(scores)
            optimal_thresholds[col] = thresholds[best_idx]
            optimal_scores[col] = scores[best_idx]
            
            # Plot threshold vs score for this class
            plt.plot(thresholds, scores, marker='o', label=f"{col} (best={thresholds[best_idx]:.2f})")
        
        # Finish and save the plot
        plt.xlabel('Threshold')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Optimal Threshold Selection by {metric.capitalize()} Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        if output_name:
            # Create the results directory if it doesn't exist
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
                
            plt.savefig(os.path.join(self.results_dir, f"{output_name}_threshold_optimization.png"))
        
        plt.close()
        
        # Calculate mean optimal threshold
        mean_threshold = np.mean(list(optimal_thresholds.values()))
        
        print("Optimal thresholds by class:")
        for col, threshold in optimal_thresholds.items():
            print(f"{col}: {threshold:.4f} ({metric}={optimal_scores[col]:.4f})")
        print(f"Mean optimal threshold: {mean_threshold:.4f}")
        
        return {
            'thresholds': optimal_thresholds,
            'scores': optimal_scores,
            'mean_threshold': mean_threshold
        }