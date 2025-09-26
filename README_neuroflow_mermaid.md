# NeuroFlow Demo - Mermaid Diagrams Documentation

## 📋 Overview

This documentation provides comprehensive mermaid diagrams for the `neuroflow_demo.ipynb` notebook, which demonstrates catastrophe detection via Neural Navier-Stokes analysis in neural networks.

## 📁 Files Created

1. **`neuroflow_demo_mermaid.md`** - Complete comprehensive diagrams
2. **`neuroflow_demo_compact.md`** - Simplified compact version
3. **`neuroflow_mermaid_viewer.html`** - Interactive HTML viewer
4. **`README_neuroflow_mermaid.md`** - This documentation file

## 🎯 Purpose

The mermaid diagrams visualize the complex workflow and architecture of the NeuroFlow system, which includes:

- **Data Generation**: Triple-point synthetic data and MNIST dataset preparation
- **Neural Network Architecture**: SimpleNet and CNN models with comprehensive monitoring
- **NeuroFlow Engine**: Advanced flow analysis system with multiple diagnostic components
- **Adversarial Testing**: 8 different attack methods for robustness testing
- **Visualization**: 3D turbulence landscapes and statistical analysis

## 🔍 Key Components Visualized

### 1. Data Processing Pipeline
- Triple-point data generation for catastrophe scenarios
- MNIST dataset loading and preprocessing
- DataLoader preparation and batch processing

### 2. Neural Network Architecture
- SimpleNet definition with Conv2d and FC layers
- Model initialization and weight loading
- Forward pass and activation capture

### 3. NeuroFlow Engine System
- **ActivationRecorder**: Captures intermediate layer activations
- **ChainMapHomology**: Detects structural dead zones using SVD
- **DMDFlowAnalyzer**: Dynamic Mode Decomposition for flow estimation
- **SpectralAnalyzer**: Graph-based turbulence analysis
- **CatastropheDetector**: Identifies decision boundary singularities

### 4. Adversarial Attack Arsenal
- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **BIM**: Basic Iterative Method
- **MI-FGSM**: Momentum Iterative FGSM
- **C&W**: Carlini & Wagner attack
- **DeepFool**: Minimal perturbation attack
- **JSMA**: Jacobian-based Saliency Map Attack
- **Square Attack**: Black-box random search

### 5. Analysis and Visualization
- PCA transformation for dimensionality reduction
- Turbulence landscape computation
- 3D visualization of catastrophe mountains
- 2D contour maps with level-set analysis
- Statistical comparison of attack methods

## 📊 Diagram Types

### Complete Architecture Diagram
Shows all system components and their relationships with color-coded subsystems:
- 🔵 Data Generation (Blue)
- 🟣 Neural Network (Purple) 
- 🟢 NeuroFlow Engine (Green)
- 🟠 Training System (Orange)
- 🟡 Analysis Components (Pink)
- 🔴 Attack Methods (Red)
- 🟢 Visualization (Light Green)

### Workflow Flow Diagram
Illustrates the step-by-step execution process:
1. Data generation and preparation
2. Model definition and initialization
3. NeuroFlow engine setup
4. Training with diagnostic monitoring
5. Adversarial attack generation
6. Analysis and visualization

### Component Interaction Diagram
Shows data flow between major system components:
- Input data sources
- Model processing pipeline
- NeuroFlow analysis components
- Adversarial testing framework
- Output visualization system

### Compact Simplified View
Focuses on main workflow steps without overwhelming detail:
- High-level process flow
- Grouped component functionality
- Key output generation

## 🚀 Usage Instructions

### Viewing the Diagrams

1. **Markdown Files**: Open `neuroflow_demo_mermaid.md` or `neuroflow_demo_compact.md` in any markdown viewer that supports mermaid (GitHub, VS Code, etc.)

2. **HTML Viewer**: Open `neuroflow_mermaid_viewer.html` in a web browser for an interactive tabbed interface

3. **Online Mermaid Editors**: Copy diagram code to:
   - [Mermaid Live Editor](https://mermaid.live/)
   - [Mermaid Chart](https://www.mermaidchart.com/)

### Integration with Documentation

These diagrams can be embedded in:
- GitHub README files
- Project documentation
- Research papers
- Presentation slides
- Technical reports

## 🔧 Technical Details

### Mermaid Syntax Features Used
- **Flowcharts** (`flowchart TD`, `graph TB`, `graph LR`)
- **Subgraphs** for logical component grouping
- **Node shapes** (rectangles, circles, diamonds, databases)
- **Edge types** (solid, dashed, dotted)
- **Styling classes** for color coordination
- **Mind maps** for hierarchical overview

### Color Coding System
- **Data/Input**: Light blue (#e1f5fe)
- **Neural Networks**: Light purple (#f3e5f5)
- **NeuroFlow Engine**: Light green (#e8f5e8)
- **Training**: Light orange (#fff3e0)
- **Analysis**: Light pink (#fce4ec)
- **Attacks**: Light red (#ffebee)
- **Visualization**: Light green (#f1f8e9)

## 📈 Benefits

1. **Visual Understanding**: Complex system architecture made comprehensible
2. **Documentation**: Self-documenting code structure
3. **Communication**: Easy to share with stakeholders
4. **Debugging**: Helps identify data flow issues
5. **Onboarding**: New team members can quickly understand the system

## 🔄 Maintenance

To update the diagrams when the notebook changes:
1. Analyze new components or workflow changes
2. Update the appropriate mermaid diagram sections
3. Test diagram rendering in mermaid editor
4. Update this documentation
5. Regenerate HTML viewer if needed

## 📞 Support

For questions about the diagrams or to request modifications:
- Check the original `neuroflow_demo.ipynb` for implementation details
- Refer to [Mermaid documentation](https://mermaid-js.github.io/mermaid/) for syntax help
- Use online mermaid editors for testing changes

---

*Generated for the Structure Net project's NeuroFlow demonstration system.*