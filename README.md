# 🔬 Hyperspectral Image Classification using SSGRN (Spectral-Spatial Graph Reasoning Network)

This repo contains an end-to-end implementation of a **Spectral-Spatial Graph Reasoning Network (SSGRN)** for hyperspectral image classification, focused on the **Indian Pines** dataset. The model captures complex relationships between pixels across both spectral and spatial dimensions using two separate GCN branches fused into a unified prediction pipeline.

## 🚀 Highlights

- 🛰️ **Dataset:** Indian Pines Hyperspectral Dataset (AVIRIS sensor, 145×145, 220 bands)
- 🧠 **Model:** Dual-branch SSGRN in PyTorch
- 🧱 **Input:** Patch-based sampling for spatial context + full-spectrum features
- 🌐 **Graph Branches:**
  - **Spectral Graph**: Correlation-based adjacency matrix in spectral domain
  - **Spatial Graph**: KNN or distance-based adjacency between pixels in patch
- ⚙️ **Fusion Strategy:** Feature fusion after parallel GCN reasoning
- 📈 **Performance Metrics:** OA, AA, F1-score, Kappa

---

## 📁 Project Structure

📦SSGRN/
📦GCN/<br>
├── indian_pines_corrected.mat # HSI Indian Pines Dataset<br>
├── indian_pines_gt.mat # ground truth<br>
├── SSGRN2.ipynb # Training + Evaluation pipeline <br>
├── best_ssgrn_model.pth # model<br>
└── README.md # You're here :)<br>


---

## 🧠 Methodology

### 🧬 What is SSGRN?

SSGRN combines two GCNs:

- **Spectral Branch:** Learns correlations across spectral bands using a graph where each node = a band.
- **Spatial Branch:** Learns relationships between neighboring pixels using a KNN-based graph.
- **Fusion:** Combines both embeddings for final classification using a shared dense layer.

### 🏗️ Architecture Overview

```text
Input Patch --> Spectral GCN -->|
                                |--> Feature Fusion --> FC Layer --> Output
Input Patch --> Spatial GCN -->|
```
### 🛠️ Installation
```bash
git clone https://github.com/Ayush-Debnath/SSGRN.git
cd SSGRN
```
### 🧪 How to Run
Download the Indian Pines .mat file and put it in the SSGRN2.ipynb file in the  having the code:
```bash
mat = sio.loadmat('/content/drive/MyDrive/Indian_pines_corrected.mat')
img = mat['indian_pines_corrected'].astype(np.float32)   # H×W×B
gt  = sio.loadmat('/content/drive/MyDrive/Indian_pines_gt.mat')['indian_pines_gt']
H, W, B = img.shape
```
Run the cells

### 📊 Sample Results
Metric	Value<br>
Overall Acc:&nbsp;&nbsp; 0.9371<br>
Average Acc:&nbsp;&nbsp;	0.9733<br>
F1 Score:&nbsp;&nbsp;	0.9373<br>
Kappa:	&nbsp;&nbsp;0.9281<br>

### 📷 Visualizations
Confusion Matrix
![image](https://github.com/user-attachments/assets/f3f80848-eb56-47bf-b89a-ea8068f71e13)

Prediction<br>
![image](https://github.com/user-attachments/assets/aa65ca7b-fcab-4aff-a310-87f2132f573f)

Accuracy Curve
![image](https://github.com/user-attachments/assets/2ba36949-22b7-4304-b6e5-71c850a87ad8)

🤝 Acknowledgements<br>
Indian Pines dataset: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines

Reasearch paper: https://ieeexplore.ieee.org/document/10234379


## 🧠 Author
👨‍💻 Ayush Debnath<br>
Engineering student, AI/ML, data scientist in the making
