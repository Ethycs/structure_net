# NeuroFlow Demo - Mermaid Diagram

This mermaid diagram represents the complete workflow and system architecture of the `neuroflow_demo.ipynb` notebook, which demonstrates catastrophe detection via Neural Navier-Stokes analysis.

## Complete System Architecture

```mermaid
graph TB
    %% Data Generation Layer
    subgraph "📊 Data Generation & Preparation"
        A1[Triple-Point Data Generator]
        A2[Synthetic Dataset Creation]
        A3[DataLoader Preparation]
        A4[MNIST Dataset Loading]
        
        A1 --> A2
        A2 --> A3
        A4 --> A3
    end

    %% Neural Network Architecture
    subgraph "🧠 Neural Network Architecture"
        B1[SimpleNet Definition]
        B2[Conv2d Layers]
        B3[FC Layers: fc1, fc2]
        B4[Output Layer]
        B5[Model Initialization]
        
        B1 --> B2
        B1 --> B3
        B1 --> B4
        B5 --> B1
    end

    %% NeuroFlow Engine System
    subgraph "🌊 NeuroFlow Engine System"
        C1[NeuroFlowEngine]
        C2[ActivationRecorder]
        C3[ChainMapHomology]
        C4[DMDFlowAnalyzer]
        C5[GraphConstructor]
        C6[SpectralAnalyzer]
        C7[LNNResidualEvaluator]
        C8[CatastropheDetector]
        
        C1 --> C2
        C1 --> C3
        C1 --> C4
        C1 --> C5
        C1 --> C6
        C1 --> C7
        C1 --> C8
    end

    %% Training and Monitoring
    subgraph "🚀 Training & Monitoring System"
        D1[Training Loop]
        D2[Activation Recording]
        D3[NeuroFlow Loss Computation]
        D4[Backpropagation]
        D5[Performance Metrics]
        
        D1 --> D2
        D2 --> D3
        D3 --> D4
        D4 --> D5
    end

    %% Analysis Components
    subgraph "🔬 Analysis Components"
        E1[Homology Loss Analysis]
        E2[DMD Flow Analysis]
        E3[Spectral Analysis]
        E4[Turbulence Computation]
        E5[Dead Zone Detection]
        E6[Catastrophe Mapping]
        
        E1 --> E5
        E2 --> E4
        E3 --> E6
        E4 --> E6
        E5 --> E6
    end

    %% Adversarial Attack System
    subgraph "⚔️ Extended Adversarial Arsenal"
        F1[ExtendedAdversarialArsenal]
        F2[FGSM Attack]
        F3[PGD Attack]
        F4[BIM Attack]
        F5[MI-FGSM Attack]
        F6[C&W Attack]
        F7[DeepFool Attack]
        F8[JSMA Attack]
        F9[Square Attack]
        
        F1 --> F2
        F1 --> F3
        F1 --> F4
        F1 --> F5
        F1 --> F6
        F1 --> F7
        F1 --> F8
        F1 --> F9
    end

    %% Visualization System
    subgraph "📈 Visualization & Analysis"
        G1[PCA Transformation]
        G2[Turbulence Landscape]
        G3[Attack Trajectory Analysis]
        G4[3D Visualization]
        G5[2D Contour Plots]
        G6[Statistical Analysis]
        
        G1 --> G2
        G2 --> G3
        G3 --> G4
        G3 --> G5
        G3 --> G6
    end

    %% Flow Connections
    A3 --> B5
    B5 --> C1
    C1 --> D1
    D1 --> E1
    E1 --> F1
    F1 --> G1
    
    %% Cross-component connections
    C2 --> D2
    C3 --> E1
    C4 --> E2
    C6 --> E3
    C8 --> E6
    
    %% Analysis to visualization
    E6 --> G2
    F2 --> G3
    F3 --> G3
    F4 --> G3
    F5 --> G3
    F6 --> G3
    F7 --> G3
    F8 --> G3
    F9 --> G3

    %% Styling
    classDef dataClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef neuralClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef engineClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef trainingClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef analysisClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef attackClass fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef vizClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class A1,A2,A3,A4 dataClass
    class B1,B2,B3,B4,B5 neuralClass
    class C1,C2,C3,C4,C5,C6,C7,C8 engineClass
    class D1,D2,D3,D4,D5 trainingClass
    class E1,E2,E3,E4,E5,E6 analysisClass
    class F1,F2,F3,F4,F5,F6,F7,F8,F9 attackClass
    class G1,G2,G3,G4,G5,G6 vizClass
```

## Detailed Flow Analysis

```mermaid
flowchart LR
    %% Main Process Flow
    subgraph "Main Processing Pipeline"
        START([Start Demo]) --> DATA[Generate Triple-Point Data]
        DATA --> MODEL[Define & Initialize SimpleNet]
        MODEL --> PRETRAIN{Load Pretrained Weights?}
        PRETRAIN -->|Yes| LOAD[Load Model Weights]
        PRETRAIN -->|No| EVAL[Evaluate Model]
        LOAD --> EVAL
        EVAL --> DMD[Setup DMD Analysis]
        DMD --> VORTICITY[Compute Vorticity]
        VORTICITY --> HOMOLOGY[Chain Map Homology]
        HOMOLOGY --> CATASTROPHE[Detect Catastrophes]
        CATASTROPHE --> NEUROFLOW[Initialize NeuroFlow Engine]
        NEUROFLOW --> TRAIN[Training with NeuroFlow]
        TRAIN --> ATTACKS[Run 8 Attack Methods]
        ATTACKS --> PCA[PCA Analysis]
        PCA --> VISUALIZE[Create Visualizations]
        VISUALIZE --> END([Complete Analysis])
    end

    %% NeuroFlow Components Detail
    subgraph "NeuroFlow Internal Process"
        NF1[Record Activations] --> NF2[Compute Homology Loss]
        NF2 --> NF3[DMD Flow Analysis]
        NF3 --> NF4[Spectral Analysis]
        NF4 --> NF5[LNN Residual Evaluation]
        NF5 --> NF6[Catastrophe Detection]
        NF6 --> NF7[Combined Loss Computation]
    end

    %% Attack Methods Detail
    subgraph "Attack Methods Pipeline"
        ATK1[FGSM] --> TRAJ[Trajectory Analysis]
        ATK2[PGD] --> TRAJ
        ATK3[BIM] --> TRAJ
        ATK4[MI-FGSM] --> TRAJ
        ATK5[C&W] --> TRAJ
        ATK6[DeepFool] --> TRAJ
        ATK7[JSMA] --> TRAJ
        ATK8[Square Attack] --> TRAJ
        TRAJ --> TURB[Turbulence Mapping]
    end

    %% Connect main flow to details
    NEUROFLOW -.-> NF1
    ATTACKS -.-> ATK1
    ATTACKS -.-> ATK2
    ATTACKS -.-> ATK3
    ATTACKS -.-> ATK4
    ATTACKS -.-> ATK5
    ATTACKS -.-> ATK6
    ATTACKS -.-> ATK7
    ATTACKS -.-> ATK8
```

## Component Interaction Diagram

```mermaid
graph LR
    %% Core Data Flow
    subgraph "Data Flow"
        D1[(Triple-Point Data)]
        D2[(MNIST Data)]
        D3[(Synthetic Data)]
    end

    subgraph "Model Processing"
        M1[SimpleNet/CNN Model]
        M2[Forward Pass]
        M3[Activation Capture]
    end

    subgraph "NeuroFlow Analysis"
        N1[ActivationRecorder]
        N2[ChainMapHomology]
        N3[DMDFlowAnalyzer]
        N4[SpectralAnalyzer]
        N5[CatastropheDetector]
    end

    subgraph "Adversarial Testing"
        A1[Attack Generation]
        A2[Trajectory Tracking]
        A3[Success Analysis]
    end

    subgraph "Visualization Output"
        V1[3D Turbulence Plot]
        V2[2D Contour Maps]
        V3[Statistical Reports]
    end

    %% Connections
    D1 --> M1
    D2 --> M1
    D3 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> N1
    N1 --> N2
    N1 --> N3
    N1 --> N4
    N2 --> N5
    N3 --> N5
    N4 --> N5
    M1 --> A1
    A1 --> A2
    A2 --> A3
    N5 --> V1
    A3 --> V1
    A3 --> V2
    A3 --> V3
```

## Key Features and Capabilities

### 🔬 Scientific Analysis Components
- **Chain Map Homology**: Detects structural dead zones using SVD analysis
- **Dynamic Mode Decomposition (DMD)**: Estimates flow field evolution
- **Spectral Analysis**: Graph-based turbulence detection
- **Catastrophe Detection**: Identifies decision boundary singularities

### ⚔️ Adversarial Attack Arsenal
- **8 Different Attack Methods**: FGSM, PGD, BIM, MI-FGSM, C&W, DeepFool, JSMA, Square Attack
- **Trajectory Analysis**: Tracks attack paths in activation space
- **Universal Landscape**: Demonstrates all attacks climb the same catastrophe mountains

### 📊 Visualization Capabilities
- **3D Turbulence Landscapes**: Interactive visualization of catastrophe regions
- **PCA-based Analysis**: Dimensionality reduction for interpretability
- **Statistical Comparisons**: Quantitative analysis of attack effectiveness
- **Contour Maps**: Level-set visualization of turbulence regions

This diagram represents the complete architecture and workflow of the NeuroFlow demonstration system for catastrophe detection in neural networks.