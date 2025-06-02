# Maze Game Using Hand Gestures

## 📑 Table of Contents

- [About the Project](#about-the-project)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Experimentation with MLflow](#experimentation-with-mlflow)
- [Model Selection & Results](#model-selection--results)
- [Staging vs Production Justification](#staging-vs-production-justification)
  

---

## 📌 About the Branch

This branch focuses on research and experimentation with various machine learning models using the HaGRID Dataset. It demonstrates a full MLOps workflow for a hand gesture recognition task, including:

- Model experimentation with scikit-learn

- MLflow tracking: parameters, metrics, models, input/output schema

- Model versioning and lifecycle management

- Environment and dependency management with venv

- Separation of research code in the research branch

**Dataset:** Synthetic bank customer data with features such as age, balance, credit score, etc.

---

## 📁 Project Structure
```bash
```


---

## ⚙️ Setup Instructions

1. **Clone and switch to the research branch**

```bash
git clone https://github.com/SaraElwatany/Maze-Game.git
cd Maze-Game
git checkout gesture-ml-production
```


2. **Create and activate virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```


3. **Install dependencies**

```bash
pip install -r requirements.txt
```


4. **Run training script**
   
```bash
python src/main.py
```



---

## 🔬Experimentation with MLflow

Multiple experiments were run and tracked:

- Model type and hyperparameters

- Accuracy, precision, recall, and F1-score

- Confusion matrix visualizations

- Input and output schema using mlflow.models.infer_signature


Tracked models:

- K-Nearest Neighbours - KNN
  
- Support Vector Machines with Linear Kernel - SVM (Linear)
  
- Support Vector Machines with RBF Kernel - SVM (RBF)
  
- Support Vector Machines with Polynomial Kernel - SVM (Poly)

- Random Forest
  
- Extreme Gradient Boosting
  


---

### 📊 Model Performance Summary

| Model                      | Hyperparameters                                                                 | Accuracy (%) | Precision | Recall  | F1-score |
|----------------------------|----------------------------------------------------------------------------------|--------------|-----------|---------|----------|
| **K-Nearest Neighbors**    | `n_neighbors=4`, `weights=distance`, `p=2`                                       | 92.54        | 0.9262    | 0.9254  | 0.9257   |
| **SVM (Linear)**           | `kernel=linear`, `C=2`, `loss=hinge`, `max_iter=3000`, `multi_class=ovr`        | 81.36        | 0.8143    | 0.8136  | 0.8120   |
| **SVM (RBF Kernel)**       | `kernel=rbf`, `C=370`, `gamma=0.5`, `decision_function_shape=ovr`               | 97.27        | 0.9728    | 0.9727  | 0.9727   |
| **SVM (Polynomial Kernel)**| `kernel=poly`, `C=2`, `gamma=10`, `degree=3`, `decision_function_shape=ovr`     | 97.80        | 0.9782    | 0.9780  | 0.9780   |
| **Random Forest**          | `n_estimators=500`                                                              | 95.09        | 0.9514    | 0.9509  | 0.9510   |
| **Extreme Gradient Boosting** | `n_estimators=500`, `learning_rate=0.1`, `max_depth=3`                      | *Pending*    | *Pending* | *Pending* | *Pending* |


**Key insights after 25+ runs:**

- **Extreme Gradient Boosting** consistently achieved the highest accuracy and balanced precision, recall, and F1 scores, making it the best candidate for production deployment due to its robust performance.

- **Random Forest** showed slightly lower metrics but offers faster computation and lower resource use, making it suitable for staging or speed-critical scenarios.



---

## 🚦 Staging vs Production Justification

To determine the most suitable models for deployment, a majority voting ensemble strategy was employed. This approach combines the predictions of multiple top-performing models to make a final decision based on the majority class predicted by those models. Specifically, the following models participated in the ensemble:

- Extreme Gradient Boosting

- Support Vector Machines (RBF Kernel)

- Support Vector Machines (Polynomial Kernel)

These models consistently performed well across key metrics (accuracy, precision, recall, F1-score), and their diversity in algorithmic approach improved generalization through ensemble voting.

While the Random Forest model also demonstrated solid performance during experimentation, it was ultimately excluded from the final ensemble due to its large model size and memory footprint, which posed challenges for deployment in resource-constrained environments (e.g., edge devices or cloud-based inference with limited capacity). The tradeoff favored maintaining a lightweight and efficient deployment pipeline without significantly compromising accuracy.

### ✅ Final Production Models:

- Extreme Gradient Boosting

- SVM with RBF Kernel

- SVM with Polynomial Kernel

These models now serve in production through an ensemble voting mechanism, ensuring robustness and reliability during inference.




