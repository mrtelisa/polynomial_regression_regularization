# Polynomial Regression and Regularization

Study of polynomial regression on 1-D toy problems in Python (NumPy): how the polynomial degree, the noise, the number of samples and the regularization affect what a model learns from data.

> First lab of the Machine Learning and Deep Learning course (2nd year, 1st semester), MSc Robotics Engineering, University of Genoa.

## Method

- **Data:** samples drawn from a known function (*oracle*) and corrupted by Gaussian noise. Since the oracle is known, the true error can be estimated on a dense test set.
- **Model:** polynomial of degree *p*, fitted by least squares, solving the normal equations directly.
- **Regularization:** the loss becomes |Xc − y|² + λ c'Mc, with two choices of M:
  - identity, i.e. ridge regression (λ|c|²);
  - a matrix that penalizes the curvature of the fitted function (integral of f''²).
- **Model selection:** grid search over the regularizer, *p* and λ on a learning/validation split.

## Notebooks

| Notebook | Content |
|---|---|
| `main1.ipynb` | Environment check |
| `main2.ipynb` | Least-squares polynomial fit of a quadratic oracle, test error, effect of high degree and few samples |
| `main3.ipynb` | Test error as a function of *p* for different noise levels: the best model is not always the "true" one |
| `main4.ipynb` | Ridge regularization: effect of λ on a degree-10 polynomial |
| `main5.ipynb` | Curvature regularization on a piecewise-linear oracle: high-degree polynomials can still give simple models |
| `main6.ipynb` | Automatic choice of regularizer, *p* and λ with a validation set, on three oracles (quadratic, piecewise-linear, sinc) |

Each notebook contains comments on the results and the plots of the learnt models.

## Usage

The notebooks were written in **Google Colab**. To run them locally:

```bash
pip install numpy matplotlib scikit-learn jupyter
jupyter notebook
```
