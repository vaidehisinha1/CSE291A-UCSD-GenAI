# CSE291A-UCSD-GenAI

This repository contains Jupyter Notebooks for exploring Generative AI in medical diagnostics, specifically using chest X-ray images for the detection of pneumonia. It covers state-of-the-art generative modeling (GANs), classifier training, and explainability methods, with thorough experimentation including ablation studies on model architecture.

---

## Repository Structure & File Overview

- **ACGAN.ipynb**: Implements an Auxiliary Classifier GAN for generating realistic chest X-ray images conditioned on "NORMAL" or "PNEUMONIA" labels. The notebook walks through data loading, model construction (generator/discriminator), training, and visualization of results.

- **Ablation1_ACGAN.ipynb**, **Ablation2_ACGAN.ipynb**, **Ablation3_ACGAN.ipynb**: These notebooks perform targeted ablation studies, each modifying one aspect of the ACGAN architecture (normalization, activation functions, or upsampling technique) to measure their impact on generated image quality and training dynamics.

- **VGGClassifier_Real.ipynb**: Trains a VGG16-based convolutional neural network to classify real chest X-ray images. Includes performance evaluation (accuracy, precision, recall, F1-score, confusion matrix), class balancing experiments, and comprehensive metrics visualization.

- **VGGClassifier_Synthetic.ipynb**: Replicates the VGG classifier setup but trains on synthetic images produced by the GAN. Results allow comparison of classifier performance between real and generated data.

- **Explainability.ipynb**: Connects generative and classifier models with state-of-the-art explainability tools (GradCAM and BLIP). It visualizes model reasoning during prediction, interprets generated image regions, and includes functionality for potential language-image explanations.
---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vaidehisinha1/CSE291A-UCSD-GenAI.git
   cd CSE291A-UCSD-GenAI
   ```

2. **Install requirements** (see the top sections of each notebook for Python package dependencies; primarily TensorFlow, Keras, PyTorch, OpenCV, scikit-learn).

3. **Obtain the Chest X-ray Dataset**:
   - Follow instructions in the notebooks for downloading the pneumonia dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

4. **Launch Jupyter Lab**:
   ```bash
   jupyter lab
   ```
   Open and run notebooks to begin experiments.

---

## Contributing

- Fork the repository.
- Create a new branch for your changes.
- Open a Pull Request describing your additions (experiments, new explainability techniques, etc.).

---

## License

MIT License (see [LICENSE](LICENSE)).

---

## Contact

- GitHub: [vaidehisinha1](https://github.com/vaidehisinha1)
- For questions about the course or repository, raise an issue or discussion!

---

## Acknowledgements

- UCSD CSE291A course, GenAI module
- Kaggle Pneumonia Chest X-ray dataset
- Frameworks: TensorFlow, Keras, PyTorch, sklearn

---

This repository is a research and teaching tool for advanced topics in generative medical AI, model interpretability, and robust classifier design.
