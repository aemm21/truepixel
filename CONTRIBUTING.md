🤝 Contributing to TruePixel

First of all, thank you for taking the time to contribute to digital transparency and truth! TruePixel is a community-driven effort focused on democratizing digital forensic verification tools.

To maintain code quality, readability, and scientific rigor, we kindly ask you to follow these guidelines.

👉 Versión en Español de este documento

🛠️ Pull Request Workflow

Fork the repository to your personal account.

Create a feature branch (git checkout -b feature/new-spectral-improvement).

Write and document your code. Ensure compatibility with PyTorch and do not break modular processes.

Test locally: Ensure that train.py (using your local dataset) and api.py run without errors in your local virtual environment.

Commit your changes with a clear and descriptive message (git commit -m 'feat: optimize Fourier branch using spectral windows').

Push to your branch (git push origin feature/new-spectral-improvement).

Open a Pull Request (PR) to TruePixel's main branch, explaining the changes made in detail.

📏 Code Style and Standards

To ensure developers from around the world can collaborate without friction, we aim to follow these standards:

Pythonic Code: General adherence to the PEP 8 style guide.

Modular Machine Learning: Do not mix local data processing with the model's architecture. Keep model.py, fourier.py, and dataset.py as pure and clean modules.

Code Comments: Write clear comments on critical mathematical functions to explain the why behind operations (especially spectral transformations and normalization steps).

Training Weights: Please do not upload heavy model weights (.pth or .pt files) in your Pull Requests. Optimized weights trained on large datasets should be distributed via GitHub "Releases" or model hosting platforms like Hugging Face.

🐞 Reporting Bugs or Proposing Ideas

If you found a bug in the detection of a newly released generative model (such as new versions of Midjourney) or have a mathematical suggestion to improve the engine:

Head over to the Issues tab in the repository.

Use a descriptive template: specify the expected behavior, the observed behavior, and provide a sample image if possible.

We greatly appreciate your dedication to building an internet free of visual manipulation!