### Naive Bayes

#### Overview

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes's theorem with the "naive" assumption of conditional independence between every pair of features given the value of the class variable. Bayes' theorem states the following relationship, given class variable $y​$ and dependent feature vector $x_1​$ through $x_n​$

$P(y|x_1, x_2, ..., x_n) = \frac {P(y) P(x_1, ..., x_n | y)}{P(x_1, ..., x_n)}$

Using the naive conditional independence assumption that 

$P(x_i | y, x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) = P(x_i|y)$

for all $i$, this relationship is simplified to 

$P(y| x_1, ..., x_n) = \frac {P(y) \prod_{i=1}^n P(x_i|y)}{P(x_1, ..., x_n)}​$

Since $P(x_1, ..., x_n)$ is constant given the input, we can use the following classification rule:

$P(y|x_1, ..., x_n) \propto P(y) \prod_{i=1}^n P(x_i | y)$

$\hat{y} = arg \space max_y p(y) \prod
_{i=1} ^n P(x_i | y)$ 

and we can use Maximum A Posteriori (MAP) estimation to estimate $P(y)$ and $P(x_i |y)$; the former is then the relative frequency of class $y$ in the training set.

The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of $P(x_i) | y$.

In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. They require a small amount of training data to estimate the necessary parameters.

Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.

#### Gaussian Naive Bayes

The likelihood of the features is assumed to be Gaussian

$P(x_i | y) = \frac {1}{\sqrt{2\pi \sigma_y^2} } exp (- \frac{(x_i- \mu_y)^2}{2\sigma_y^2})$

The parameters $\sigma_y$ and $\mu_y$ are estimated using maximum likelihood.