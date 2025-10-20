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

Lecturer: **Oualid Missaoui** — reach out at `oualid.missaoui@baruch.cuny.edu` or during office hours.
