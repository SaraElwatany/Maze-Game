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

We monitor **four key metrics** using Prometheus and visualize them with Grafana:

1. **🔁 Total Model Fails`model_failures_total`**  
   - **What it measures**: Total number of failed model inferences (e.g., internal server errors during prediction).  
   - **Type**: Counter  
   - **PromQL Query**:  
     ```promql
     model_failures_total
     ```  
   - **Grafana Visualizations**:  
     - Stat panel (shows current failure count)  

2. **🌐HTTP Requests Per Route `sum by (endpoint) (http_requests_total)`**  
   - **What it measures**: Total number of HTTP requests per endpoint since the app started.  
   - **Type**: Counter  
   - **PromQL Query**:  
     ```promql
     sum by (endpoint) (http_requests_total)
     ```  
   - **Grafana Visualizations**:  
     - Bar chart (for endpoint comparison)  

3. **⏱️ Average Latency**  
   - **What it measures**: Average inference latency per request over the last minute.  
   - **Type**: Gauge  
   - **PromQL Query**:  
     ```promql
     rate(model_inference_time_seconds_sum[1m]) / rate(model_inference_time_seconds_count[1m])
     ```  
   - **Grafana Visualizations**:  
     - Time series (to monitor changes over time)  

4. **📥 Invalid Data Input`rate(data_input_errors_total[1m])`**  
   - **What it measures**: Rate of invalid or malformed input data received in the last minute.  
   - **Type**: Counter (rate-wrapped)  
   - **PromQL Query**:  
     ```promql
     rate(data_input_errors_total[1m])
     ```  
   - **Grafana Visualizations**:  
     - Time series (to identify spikes or trends)  


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
├── .github/workflows/         # GitHub Actions workflows for continuous deployment
│ └── cicd.yml               # Deployment pipeline configuration
├── grafana/provisioning/dashboards/     # Pre-configured Grafana dashboards
├── models/                              # Directory for trained model files
├── prometheus/                          # Prometheus monitoring configuration
├── testing/                             # Testing scripts and utilities
├── .gitattributes                       # Git LFS and file-type tracking rules
├── .gitignore                           # Files and folders ignored by Git
├── Dockerfile                           # Docker image configuration
├── README.md                            # Project overview and documentation
├── apis.py                              # Main API script (e.g., for serving model)
├── docker-compose.yaml                  # Orchestrates services (API, Prometheus, Grafana)
├── download_model.py                    # Script for fetching/downloading the model
├── preprocessing.py                     # Data preprocessing utilities
├── requirements.txt                     # Python dependencies for the project
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

**Below are the selected metrics with justifications:**

---

#### ✅ 1. Model-Related Metric: `model_failures_total`

- **Why it matters**: Tracks the **total number of failed inferences**, such as internal errors during model execution.
- **Use case**: Identifies instability in the model's behavior and helps diagnose recurring prediction failures.
- **Production benefit**: Ensures the model remains **reliable and trustworthy** in real-world usage.

---

#### ✅ 2. Data-Related Metric: `rate(data_input_errors_total[1m])`

- **Why it matters**: Monitors the **rate of malformed or invalid input data** sent to the model.
- **Use case**: Detects **data drift**, input format changes, or upstream pipeline issues.
- **Production benefit**: Protects the model from being fed poor-quality data, which could degrade performance.

---

#### ✅ 3. Server-Related Metric: `rate(model_inference_time_seconds_sum[1m]) / rate(model_inference_time_seconds_count[1m])`

- **Why it matters**: Measures **average inference time** per request over the last minute.
- **Use case**: Detects **backend latency** and helps maintain a responsive user experience.
- **Production benefit**: Ensures the API is **fast and scalable**, especially under load.

---

#### ✅ 4. API Usage Metric: `sum by (endpoint) (http_requests_total)`

- **Why it matters**: Shows how often each **API endpoint** is accessed.
- **Use case**: Identifies **popular endpoints**, unused routes, and potential abuse patterns.
- **Production benefit**: Informs optimization and scaling decisions based on real user behavior.




