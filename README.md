# SVM VisionLab: Image-Based Object Recognition using SVM

A complete, full-stack **Python & Scikit-Learn** web application and machine learning engine for **Image-Based Object Recognition using Support Vector Machines (SVM)**.

---

## Key Features

1. **Computer Vision & Feature Engineering Pipelines**:
   - **Histogram of Oriented Gradients (HOG)** with configurable cell size, 9-bin orientation voting, and L2-Hys block normalization.
   - **Color Histograms & Moments** in HSV / RGB space.
   - **Raw Pixel Intensities** and spatial downsampling.
   - Combined multi-modal descriptors.

2. **Full-Stack Python Web Application**:
   - **Interactive Training Studio**: Configure datasets (Scikit-Learn Digits, Geometric Shapes, Vehicles vs. Animals), kernel types (`RBF`, `Linear`, `Polynomial`), regularization ($C$), and $\gamma$.
   - **Real-Time Evaluation Metrics**: Test Accuracy, Weighted F1-Score, Support Vector count, Confusion Matrix generation, and 2D Decision Boundary projection plots.
   - **Live Recognition Studio**: Interactive canvas drawing pad with touch/mouse support, preset image samples, and drag-and-drop file upload.
   - **Step-by-Step HOG Visualizer**: Inspect intermediate arrays (Grayscale, Sobel $G_x$, Sobel $G_y$, Gradient Magnitude, 8×8 cell grids).
   - **One-vs-Rest (OvR) Margin Scores & Probabilities**.
   - **Dynamic Python Script Generator**: Export production-ready Scikit-Learn code.

3. **Standalone CLI Training Script**:
   - Run command-line benchmarks and tests directly via `python train_offline.py`.

---

## Directory Structure

```
svm-object-recognition/
├── app.py                # Main Flask web application server & REST APIs
├── svm_engine.py         # ML backend (HOG feature extraction, SVM training, PCA boundary)
├── train_offline.py      # Standalone CLI training script
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── templates/
│   └── index.html        # Interactive Web Interface (Tailwind CSS, KaTeX, Lucide Icons)
└── static/
    ├── css/
    │   └── style.css     # Custom animations & styling
    └── js/
        └── main.js       # Drawing canvas, live inference, and AJAX communication
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Run Standalone CLI Benchmarks
```bash
# Train on Digits with RBF kernel and HOG features
python train_offline.py --dataset digits --feature hog --kernel rbf --C 1.0

# Train on Geometric Shapes with Linear kernel
python train_offline.py --dataset shapes --feature hog --kernel linear --C 2.0
```

---

## Theoretical Foundations

### 1. Support Vector Machine Optimization (Primal Soft-Margin)
Given $N$ training points $(\mathbf{x}_i, y_i)$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
$$\text{subject to } y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

### 2. The Kernel Trick (Mercer's Theorem)
Instead of explicitly computing the high-dimensional mapping $\phi(\mathbf{x})$, Mercer kernels compute inner products directly:
- **Radial Basis Function (RBF / Gaussian)**:
  $$K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$$
- **Polynomial**:
  $$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^d$$
- **Linear**:
  $$K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$$

### 3. Histogram of Oriented Gradients (HOG)
1. **Partial derivatives**: $G_x = I * [-1, 0, 1]$, $G_y = I * [-1, 0, 1]^T$
2. **Gradient magnitude & orientation**:
   $$\mu(x,y) = \sqrt{G_x^2 + G_y^2}, \quad \theta(x,y) = \arctan\left(\frac{G_y}{G_x}\right) \bmod 180^\circ$$
3. **Spatial voting**: Accumulate votes into 9 orientation bins per cell.
4. **Block normalization**: Group adjacent cells and normalize using L2-Hys clipping.
