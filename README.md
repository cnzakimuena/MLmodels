# Machine Learning Classification
Script which implements several classical machine learning models for classifiation ([Brownlee, 2023](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)). The Iris dataset ([Anderson, 1935](https://wiki.irises.org/pub/Hist/Info1986SIGNA37/SIGNA_37.pdf); [Anderson, 1936](https://doi.org/10.2307/2394164); [Fisher, 1936](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)) is used for demonstration. Logistic regression ([Cox, 1958](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x)), linear discriminant analysis ([Fisher, 1936](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)), k-nearest neighbors ([Cover & Hart, 1967](https://doi.org/10.1109%2FTIT.1967.1053964)), classification and regression trees (Breiman et al, 1984), naive Bayes ([Mosteller & Wallace, 1963](https://doi.org/10.2307/2283270)) and support-vector machine ([Cortes & Vapnik, 1995](https://doi.org/10.1023/A:1022627411411)) models are featured. A short description of each model is provided below:

- **Logistic regression (LR)** : Estimates a multiple linear regression function.
- **Linear Discriminant Analysis (LDA)** : Estimates the mean and variance from the dataset’s features for each class and uses Bayes Theorem to estimate the probability of observations belonging to classes (assumes Gaussian class-conditional density models).
- **k-Nearest Neighbors (k-NN)** : Assigns class to observations based on the selected value k of its nearest neighbors in the feature space.
- **Classification and regression trees (CART)** : Iterates across features and finds rules which best divide data into given classes along a tree structure.
- **Naive Bayes (NB)** : Uses Bays Theorem to estimate the probability of observations belonging to classes (assumes features to be independent).
- **Support-Vector Machine (SVM)** : Observations are mapped into progressively higher dimensions until a hyperplane is found which best divides the dataset into classes (support vectors are the observations nearest to the hyperplane).

Environment setup:

```bash
conda create -n myenv python=3.11
conda activate myenv
```

Dependencies installation:

```bash
pip install -r requirements.txt
```

Usage:

```bash
python classification_ml.py
```

### References

1. [Brownlee, J. Your First Machine Learning Project in Python Step-By-Step. 2019.](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
1. [Anderson, E. (1935). The irises of the Gaspe Peninsula. Bulletin of American Iris Society, 59, 2-5.](https://wiki.irises.org/pub/Hist/Info1986SIGNA37/SIGNA_37.pdf)
1. [Anderson, E. (1936). The species problem in Iris. Annals of the Missouri Botanical Garden, 23(3), 457-509.](https://doi.org/10.2307/2394164)
1. [Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of eugenics, 7(2), 179-188.](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)
1. [Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society Series B: Statistical Methodology, 20(2), 215-232.](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x)
1. [Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1), 21-27.](https://doi.org/10.1109%2FTIT.1967.1053964)
1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
1. [Mosteller, F., & Wallace, D. L. (1963). Inference in an authorship problem: A comparative study of discrimination methods applied to the authorship of the disputed Federalist Papers. Journal of the American Statistical Association, 58(302), 275-309.](https://doi.org/10.2307/2283270)
1. [Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.](https://doi.org/10.1023/A:1022627411411)