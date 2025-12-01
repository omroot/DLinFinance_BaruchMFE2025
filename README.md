# Deep Learning in Finance (Baruch MFE)

This repository hosts the lecture material, notebooks, and homework for the Baruch College MFE elective **Deep Learning in Finance (2025)**. Content will expand as the semester progresses; start by reviewing the week-by-week resources below.

## Course Syllabus
- Download the full syllabus (Fall 2025) from [`Syllabus/DeepLearningInFinance_Syllabus_BaruchMFE_Fall2025.pdf`](Syllabus/DeepLearningInFinance_Syllabus_BaruchMFE_Fall2025.pdf).

## Week 1
- Slides: download the lecture decks from [`lecture_slides/week1/`](lecture_slides/week1/).
- Theory notebooks: review the foundational materials in `src/notebooks/theory/`. Key notebooks include:
  - [Approximation.ipynb](src/notebooks/theory/approximation.ipynb)
  - [Double Descent.ipynb](src/notebooks/theory/double_descent.ipynb)
  - [Spectral Analysis.ipynb](src/notebooks/theory/spectral_analysis.ipynb)
  - [Scaling Laws.ipynb](src/notebooks/theory/scaling_laws.ipynb)
- Homework: begin with [`homeworks/week1/assignment/hw01.md`](homeworks/week1/assignment/hw01.md) and pull any supporting files from the same directory.

## Week 2
- Slides: download the lecture decks from [`lecture_slides/week2/`](lecture_slides/week2/).
- Transformer notebooks: explore transformer architectures for time series forecasting in `src/notebooks/transformer/`. Key notebooks include:
  - [Transformer Univariate Time Series Forecasting.ipynb](src/notebooks/transformer/transformer_univariate_timeseries_forecasting.ipynb)
  - [Auto-Regressive Transformer Univariate Time Series Forecasting.ipynb](src/notebooks/transformer/auto_regressive_transformer_univariate_timeseries_forecasting.ipynb)
- Homework: begin with [`homeworks/week2/assignment/hw02.md`](homeworks/week2/assignment/hw02.md) and pull any supporting files from the same directory.

## Week 3
- Slides: download the lecture decks from [`lecture_slides/week3/`](lecture_slides/week3/).
- Neural ODE notebooks: explore neural ordinary differential equations in `src/notebooks/node/`. Key notebooks include:
  - [Neural ODE Illustration.ipynb](src/notebooks/node/neural_ode_illustration.ipynb)
  - [Latent Neural ODE Illustration.ipynb](src/notebooks/node/latent_neural_ode_illustration.ipynb)
- Homework: [`homeworks/week3/homework_latent_ode_bond_trading.ipynb`](homeworks/week3/homework_latent_ode_bond_trading.ipynb)

## Week 4
- Slides: download the lecture decks from [`lecture_slides/week4/`](lecture_slides/week4/).
- GAN notebooks: explore Generative Adversarial Networks for financial applications in `src/notebooks/gan/`. Key notebooks include:
  - [Vanilla GAN Illustration.ipynb](src/notebooks/gan/vanilla_gan_illustration.ipynb)
  - [Conditional GAN Classification Illustration.ipynb](src/notebooks/gan/conditional_gan_classification_illustration.ipynb)
  - [Conditional GAN Regression Illustration.ipynb](src/notebooks/gan/conditional_gan_regression_illustration.ipynb)
  - [Wasserstein GAN Illustration.ipynb](src/notebooks/gan/wgan_illustration.ipynb)
  - [InfoGAN Illustration.ipynb](src/notebooks/gan/infogan_illustration.ipynb)
- Homework: begin with [`homeworks/week4/assignment/README.md`](homeworks/week4/assignment/README.md) for instructions on the bond pricing CGAN project.

## Week 5
- Slides: download the lecture decks from [`lecture_slides/week5/`](lecture_slides/week5/).
- Diffusion model notebooks: explore diffusion models and score-based generative models in `src/notebooks/diffusion/`. Key notebooks include:
  - [DDPM 2D Illustration.ipynb](src/notebooks/diffusion/ddpm_2d_illustration.ipynb)
  - [Score Matching Diffusion.ipynb](src/notebooks/diffusion/score_matching_diffusion.ipynb)
  - [NCSN Multiscale Score Matching.ipynb](src/notebooks/diffusion/ncsn_multiscale_score_matching.ipynb)
  - [Score SDE Diffusion.ipynb](src/notebooks/diffusion/score_sde_diffusion.ipynb)
- Homework: [`homeworks/week5/homework_ddpm_covariance_shrinkage.ipynb`](homeworks/week5/homework_ddpm_covariance_shrinkage.ipynb)

## Repository Layout
- `homeworks/`: weekly assignments and starter code.
- `lecture_slides/`: slide decks used in class.
- `src/`: Python modules and interactive notebooks (see below for details).

### `src/` structure
- `src/dlf/`: reusable course library code.
  - `models/`: experiment-ready model implementations  
  - `preprocess/`: data prep utilities.
  - `model_selection/`: training loops, evaluation helpers, and experiment orchestration.
  - `theory/`: helper functions that accompany the theoretical notebooks.
- `src/notebooks/`: illustrative notebooks used to demystify concepts discussed in class.

## Getting Started
1. Clone the repository: `git clone <repo-url>`
2. (Optional) Create and activate a virtual environment.
3. Install Python dependencies as needed for individual notebooks (environment files will be provided alongside future labs).

## Use of AI in Course Material Preparation

AI tools were used to assist in the preparation of this course material, primarily for code refactoring and polishing the presentation of assignments and illustration notebooks. The instructor wrote the original code and concepts, and AI was employed to improve code quality, consistency, and presentation.

**The following remain the sole responsibility of the instructor:**
- Selection and sequencing of topics
- Conceptual content and pedagogical approach
- Lecture structure and ordering
- Preparation of all lecture slides
- Academic integrity and accuracy of course material

Lecturer: **Oualid Missaoui** — reach out at `oualid.missaoui@baruch.cuny.edu` or during office hours.
