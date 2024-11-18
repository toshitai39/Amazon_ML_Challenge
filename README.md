# Amazon_ML_Challenge
This repository hosts the implementation of a machine learning model designed to extract entity values from images. Such a capability is invaluable across industries like healthcare, e-commerce, and content moderation, where precise and automated extraction of product details is critical.


Multimodal Entity Extraction from Images
Welcome to the Multimodal Entity Extraction repository! This project explores advanced machine learning techniques for extracting key entity values from images using large, multimodal vision-language models. Our work focuses on leveraging state-of-the-art models such as InternVL2-8B and Qwen2-VL-7B, fine-tuning their capabilities for optimal performance in real-world applications like OCR, image captioning, and visual question answering.

🌟 Key Features
Multimodal Learning: Combines visual and textual data for robust entity extraction.
State-of-the-Art Models: Utilizes InternVL2-8B and fine-tuned Qwen2-VL-7B.
Efficient Fine-Tuning: Implements LoRA (Low-Rank Adaptation) for computationally efficient model updates.
Batch Inference: Optimized for high-throughput inference with tools like LMDeploy and LLaMA-Factory.
Custom Post-Processing: Generates structured outputs (e.g., CSV) for downstream tasks.


📋 Project Overview
This repository outlines the implementation of two top-tier multimodal models:

InternVL2-8B

Architecture: Vision-Language Transformer.
Components: InternLM2.5-7B (language) and InternViT-300 (vision).
Benchmark: Achieved an OCR benchmark score of 794.
Qwen2-VL-7B

Architecture: Multimodal model using Qwen2-7B and ViT-300 as backbone.
Leaderboard: Ranked among the top models in the sub-20B category on the Open VLM leaderboard.


🚀 Approach
Pretrained Model Inference
Baseline performance was evaluated using InternVL2-8B on the test dataset.
Results guided further refinement and model selection.
Fine-Tuning Qwen2-VL-7B
Fine-tuned with LoRA for efficient optimization using:
Batch Size per Device: 2
Gradient Accumulation Steps: 4
Learning Rate: 5e-5
Epochs: 3
LoRA injected trainable low-rank matrices into attention layers, reducing memory usage while maintaining performance.
Inference and Post-Processing
LMDeploy: Enhanced inference speed and efficiency, achieving up to 1.8x higher request throughput.
Outputs were passed through a custom pipeline to generate structured CSV outputs.
⚙️ Technologies and Tools


Models:

InternVL2-8B
Qwen2-VL-7B
Optimization and Deployment:

LoRA via LLaMA-Factory
Inference Toolkit: LMDeploy


📊 Experiments and Results
Batch inferencing was conducted on the top 5 models from the Open VLM leaderboard to compare accuracy, efficiency, and adaptability.
The fine-tuned Qwen2-VL-7B achieved an F1 score of 0.704 on the test dataset.
Observed areas for improvement:
Incorporating null entity value samples during fine-tuning.
Reducing hallucinations in predictions for improved reliability.

🛠️ Setup and Usage
Prerequisites
Python 3.x
CUDA-enabled GPU (recommended)
Required libraries (listed in requirements.txt)

Installation

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/multimodal-entity-extraction.git  
cd multimodal-entity-extraction  
Install dependencies:

bash
Copy code
pip install -r requirements.txt  
Run Inference
Preprocess your dataset (refer to data_preprocessing.py).


📄 License
This project is licensed under the MIT License.

📫 Contact
For questions or feedback, reach out to us at your.email@example.com.
