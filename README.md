# 🧠 Maze Game Using Hand Gestures


## 📑 Table of Contents
- [About the Branch](#about-the-branch)
- [Project Structure](#project-structure)
- [Setup Instructions](#️-setup-instructions-for-local-running)
- [Grafana Dashboard](#grafana-dashboard)
- [Why These Metrics](#why-these-metrics)


---


## 📌 About the Branch

This branch is focused on the **production** stage of the project.

### ✅ Monitoring Metrics 
Selected and monitored three metrics:
- **Model-related**
- **Data-related**
- **Server-related**

Each metric is justified and explained in this `README.md` file under the [Why These Metrics](#why-these-metrics) section.

### ✅ System Monitoring
- Set up with **Docker Compose**
- Collect and visualize the selected metrics using **Grafana**
- A Grafana dashboard screenshot is included in the repo

### ✅ Deployment
- The deployment workflow is defined
- Application is deployed on **ClawCloud**.

### ✅ Continuous Deployment with GitHub Actions
- Configured **GitHub Actions** to automate deployment
- On every push to the `gesture-ml-production` branch:
  - Code is linted and tested
  - Docker image is built and pushed
  - App is deployed to the production server
- Ensures quick and consistent delivery of updates



---



## 📁 Project Structure
```<code>
Maze-Game/
├── mlartifacts/       # Stores serialized models, metrics, and artifacts from experiments
├── mlruns/            # MLflow tracking directory for experiment runs
├── src/               # Source code including data loading, preprocessing, and training scripts
├── plot_confusion_matrix.png       # Visualization of the confusion matrix for model evaluation
├── requirements.txt                # Python dependencies for setting up the environment
├── .gitignore                      # Files and directories to be ignored by Git
└── README.md                       # Project overview and instructions
```


---


## ⚙️ Setup Instructions (For Local Running)

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
python apis.py
```





---



## 📉 Grafana Dashboard

Metrics are visualized in Grafana and collected using Prometheus.
Docker Compose is used to manage services.

![Screenshot 2025-06-02 030124](https://github.com/user-attachments/assets/17b4f71f-2430-428e-8470-37a0042ac303)



---




## 🧩 Why These Metrics


**1. Model-Related Metric: Accuracy**

- Reflects how well the gesture recognition model performs

- Critical for determining if predictions are reliable in a production setting


**2. Data-Related Metric: Input Frequency**

- Monitors how frequently new gesture data is received

- Helps detect data drift or missing input issues
  

**3. Server-Related Metric: API Response Time**

- Measures backend latency in delivering predictions

- Ensures smooth UX and flags performance bottlenecks




