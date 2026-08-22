# AgentPINN: Multi-Agent Automated Design for PINNs 🚀

**Notice:** The code and methodology in this repository are part of the paper:
> *"A Multi-Agent Framework for Automated Design and Training of Physics-Informed Neural Networks: Application to Natural Convection in Porous Media"* 
> **Status:** Submitted to *Engineering Applications of Artificial Intelligence* (Under Review, July 2026).

---

## 📖 Overview
Architecting a Physics-Informed Neural Network (PINN) typically relies on heavy human heuristic tuning. This project introduces an autonomous, LLM-driven **Multi-Agent System** that collaboratively designs, trains, and validates PINNs without human intervention. 

Instead of blind trial-and-error, specialized agents reason about the physics, propose network architectures, execute the training loop, and use engineering solvers (like COMSOL) as **verifiable supervisors (rewards)** to validate the network's outputs.

## 🧠 The Multi-Agent Architecture
The framework is orchestrated through a role-playing multi-agent workflow:
1. **The Manager Agent:** Oversees the pipeline, defines the optimization strategy, and delegates tasks.
2. **The Mechanical Engineer Agent:** Formulates the physical equations (continuity, momentum, heat transfer in porous media) and constructs the physics-informed loss functions.
3. **The Computer Engineer Agent:** Designs the neural network architecture (layers, neurons, activation functions) and writes the PyTorch training loop.
4. **The Reporter Agent:** Evaluates the model's predictions against ground-truth/reference data and provides feedback for the next iteration.

## 🔬 Key Results
* **Accuracy:** Successfully modeled natural convection in porous media, reproducing reference COMSOL Multiphysics results with **< 1% error**.
* **Automation:** Eliminated manual hyperparameter tuning by allowing agents to autonomously iterate based on verifiable physical constraints.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch
* **Agent Orchestration:** LangGraph
* **Physics Modeling:** COMSOL Multiphysics (for reference validation)

---
*Note: Full source code will be progressively open-sourced pending the journal's peer-review process. Core agent interaction schemas and PyTorch network generators are provided as examples.*
