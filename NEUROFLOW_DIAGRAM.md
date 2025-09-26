# NeuroFlow Demo - System Architecture Diagram

This is the main mermaid diagram for the `neuroflow_demo.ipynb` notebook system.

## Complete NeuroFlow System Architecture

```mermaid
flowchart TD
    %% Main workflow steps
    A[📊 Generate Triple-Point Data] --> B[🧠 Define Neural Network]
    B --> C[💾 Load/Train Model]
    C --> D[🌊 Initialize NeuroFlow Engine]
    D --> E[🔬 Analyze Network Flow]
    E --> F[⚔️ Run 8 Attack Methods]
    F --> G[📈 Generate Visualizations]
    
    %% NeuroFlow components
    subgraph "NeuroFlow Analysis"
        E1[Homology Analysis]
        E2[DMD Flow Analysis]
        E3[Catastrophe Detection]
        E4[Turbulence Mapping]
    end
    
    %% Attack methods
    subgraph "Adversarial Attacks"
        F1[FGSM/PGD/BIM]
        F2[MI-FGSM/C&W]
        F3[DeepFool/JSMA/Square]
    end
    
    %% Visualizations
    subgraph "Output Visualizations"
        G1[3D Turbulence Landscape]
        G2[2D Contour Maps]
        G3[Statistical Analysis]
    end
    
    %% Connect subgraphs
    E --> E1
    E --> E2
    E --> E3
    E --> E4
    
    F --> F1
    F --> F2
    F --> F3
    
    G --> G1
    G --> G2
    G --> G3
    
    %% Data flow
    E4 --> G1
    F1 --> G1
    F2 --> G2
    F3 --> G3

    %% Styling
    classDef processClass fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef analysisClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef attackClass fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    classDef outputClass fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class A,B,C,D,E,F,G processClass
    class E1,E2,E3,E4 analysisClass
    class F1,F2,F3 attackClass
    class G1,G2,G3 outputClass
```

## System Components

### 📊 Data Generation & Preparation
- **Triple-Point Data Generator**: Creates synthetic dataset with catastrophe scenarios
- **MNIST Dataset Loading**: Standard benchmark dataset for neural network testing
- **DataLoader Preparation**: Batch processing and data pipeline setup

### 🧠 Neural Network Architecture
- **SimpleNet Definition**: Basic neural network architecture with Conv2d and FC layers
- **CNN Model**: Convolutional neural network for image processing
- **Model Initialization**: Weight initialization and architecture setup

### 🌊 NeuroFlow Engine System
- **ActivationRecorder**: Captures intermediate layer activations during forward passes
- **ChainMapHomology**: Detects structural dead zones using SVD analysis
- **DMDFlowAnalyzer**: Dynamic Mode Decomposition for flow field estimation
- **SpectralAnalyzer**: Graph-based turbulence analysis
- **CatastropheDetector**: Identifies decision boundary singularities

### ⚔️ Extended Adversarial Arsenal
- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **BIM**: Basic Iterative Method
- **MI-FGSM**: Momentum Iterative FGSM
- **C&W**: Carlini & Wagner attack
- **DeepFool**: Minimal perturbation attack
- **JSMA**: Jacobian-based Saliency Map Attack
- **Square Attack**: Black-box random search

### 📈 Visualization & Analysis
- **PCA Transformation**: Dimensionality reduction for activation space analysis
- **3D Turbulence Landscapes**: Interactive visualization of catastrophe regions
- **2D Contour Maps**: Level-set visualization of turbulence regions
- **Statistical Analysis**: Quantitative comparison of attack methods

## Key Features

- **Complete Flow Analysis**: From data generation to final visualization
- **8 Attack Methods**: Comprehensive adversarial testing suite
- **Scientific Approach**: Based on mathematical analysis of neural flow dynamics
- **Interactive Visualization**: 3D landscapes and statistical comparisons
- **Catastrophe Detection**: Identifies decision boundary singularities

This diagram represents the complete workflow implemented in the `neuroflow_demo.ipynb` notebook.