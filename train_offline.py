"""
Standalone SVM Image Object Recognition Training & Evaluation Script
=====================================================================
Usage:
    python train_offline.py --dataset digits --feature hog --kernel rbf --C 1.0
"""

import argparse
from svm_engine import SVMVisionEngine


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SVM on image datasets.")
    parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "shapes", "vehicles_vs_animals"],
                        help="Dataset to train on (digits, shapes, vehicles_vs_animals)")
    parser.add_argument("--feature", type=str, default="hog", choices=["hog", "raw", "color_hist", "hog_color"],
                        help="Feature extraction descriptor")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf", "poly"],
                        help="SVM Kernel type")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter C")
    parser.add_argument("--gamma", type=str, default="scale", help="Kernel coefficient gamma")
    parser.add_argument("--grid-search", action="store_true", help="Perform K-fold Grid Search CV")

    args = parser.parse_args()

    print("\n=======================================================")
    print("  SVM Image-Based Object Recognition Training")
    print("=======================================================")
    print(f" Dataset:           {args.dataset}")
    print(f" Feature Extractor: {args.feature.upper()}")
    print(f" SVM Kernel:        {args.kernel.upper()}")
    print(f" Regularizer C:     {args.C}")
    print(f" Gamma:             {args.gamma}")
    print("-------------------------------------------------------")

    engine = SVMVisionEngine()
    metrics = engine.train(
        dataset_name=args.dataset,
        feature_type=args.feature,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        use_grid_search=args.grid_search
    )

    print("\n[TRAINING & EVALUATION COMPLETE]")
    print(f" Total Samples:       {metrics['num_samples']}")
    print(f" Feature Vector Dim:  {metrics['feature_dim']}")
    print(f" Support Vectors:     {metrics['num_support_vectors']}")
    print(f" Test Accuracy:       {metrics['accuracy']}%")
    print(f" Weighted F1-Score:   {metrics['f1_score']}%")
    print(f" Training Latency:    {metrics['training_time_sec']} seconds")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
