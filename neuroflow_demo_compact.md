# NeuroFlow Demo - Compact Mermaid Diagram

A simplified version of the NeuroFlow demonstration workflow for easier viewing and understanding.

## Simplified Workflow

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

## Key Components Overview

```mermaid
mindmap
  root((NeuroFlow Demo))
    Data Generation
      Triple-Point Dataset
      MNIST Loading
      Synthetic Data
    Neural Network
      SimpleNet Architecture
      Conv2d + FC Layers
      Model Training
    NeuroFlow Engine
      Activation Recording
      Homology Analysis
      DMD Flow Analysis
      Catastrophe Detection
    Adversarial Testing
      8 Attack Methods
      Trajectory Analysis
      Success Metrics
    Visualization
      3D Landscapes
      2D Contours
      Statistical Reports
```

## System Architecture Summary

```mermaid
graph LR
    subgraph "Input Layer"
        I1[Data]
        I2[Model]
    end
    
    subgraph "Processing Core"
        P1[NeuroFlow Engine]
        P2[Attack Arsenal]
    end
    
    subgraph "Output Layer"
        O1[Analysis Results]
        O2[Visualizations]
    end
    
    I1 --> P1
    I2 --> P1
    I2 --> P2
    P1 --> O1
    P2 --> O1
    O1 --> O2
```

This compact version highlights the main workflow and key components of the NeuroFlow demonstration system without overwhelming detail.