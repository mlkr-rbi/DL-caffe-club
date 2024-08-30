# Regularization - matrix perspective

In this section we will first derive a naive solution for the Ordinary Least Squares (OLS) problem, which estimates coefficients in a linear regression model. We will see that naive solution is not always easy (or even possible) to calculate, and we will comment on the conditions under which this is the case from the perspective of spectral properties of matrices involved. In the end, we will show how to use these spectral properties to derive a new regularized solution to the OLS problem - a ridge regression, which is easier to calculate under wider range of conditions.

**Ordinary Least Squares (OLS)** is a method for estimating the unknown parameters (coefficients) in a linear regression model:

$$y = X w$$

where $y$ is the target vector, $X$ is the data matrix, and $w$ are the coefficients of our regression.

The objective function for the OLS is: 

$$L(w) = \Vert X w - y \Vert^2$$

The naive OLS solution is obtained by solving:

$$\nabla_w L(w) = 0$$

The gradient of the OLS objective function is:

$$\nabla_w L(w) = 2 X^T (X w - y)$$

We set the gradient to zero:

$$2 X^T (X w - y) = 0$$

And then we solve for $w$:

$$X^T X w = X^T y$$

$$w = (X^T X)^{-1} X^T y$$

This is the naive OLS solution - *naive* because it is not always easy (or even possible) to calculate $(X^T X)^{-1}$ (the inverse of the Gram matrix $X^T X$) in practice! The problem is that the Gram matrix $X^T X$ can be ill-conditioned or nearly singular (non-invertible) if the columns (features) of our data matrix $X$ are highly correlated (multicollienarity), or if there are more features than samples (high-dimensional data).

We will now show under which conditions is a typical matrix invertible.

**Invertibility of matrix** is a property of a square matrix $A$ which has an inverse $A^{-1}$ such that:

$$A A^{-1} = A^{-1} A = I$$

where $I$ is the identity matrix. Note that we cannot easily solve the original linear regression problem $y = X w$ by inverting matrix $X$ (to obtain a solution $w = X^{-1} y$) because matrix $X$ is typically not square! Although there are techniques for inverting non-square matrics (for example, Moore-Penrose pseudoinverse) they are essentially equally complex as inverting a (square) Gram matrix $X^T X$ and subject to same numerical considerations we will outline in the remainder of the section.

In order to know whether our Gram matrix $X^T X$ is invertible or not, we have to inspect its *spectral properties* - its set of *eigenvalues*. This will also later help us devise a method of *regularization* which will ensure that our Gram matrix $X^T X$ is well-conditioned and invertible.

**Spectral analysis** is a method of performing an *eigenvalue decomposition* of a matrix $A$:

$$A = Q \Lambda Q^T$$

where $Q$ is the orthogonal matrix of eigenvectors of $A$, and $\Lambda$ is the diagonal matrix of eigenvalues $\lambda_{i}$ of $A$.

There is a direct relationship between the eigenvalues of a matrix $A$ and the eigenvalues of its inverse $A^{-1}$ - the eigenvalues of the inverse matrix are the reciprocals of the eigenvalues of the original matrix! Therefore if $\lambda_{i}$ are the eigenvalues of $A$, then the eigenvalues of $A^{-1}$ are $\frac{1}{\lambda_{i}}$.

This already gives us a minimum spectral condition for matrix to be invertible - matrix needs to have all non-zero eigenvalues. If any of the eigenvalues of a matrix are zero, the matrix is *singular* and its inverse does not exist. Matrices with all non-zero eigenvalues are called *full rank* matrices. An existence of at least one zero eigenvalue implies that the matrix is *rank-deficient* and that at least some of its rows or columns are linearly dependent - at least one row or columns can be expressed as a linear combination of other rows or columns.

A matrix which is easiest to invert is an *orthogonal* matrix - a matrix whose columns are orthogonal to each other (dot products of its rows and columns are zero). For example, matrix $Q$ obtained from eigenvalue decomposition earlier is orthogonal. In that case the inverse of the matrix is simply its transpose:

$$Q^{-1} = Q^T$$

Eigenvalues of an orthogonal matrix are either 1 or -1, so the inverse of an orthogonal matrix is also orthogonal.

However, a matrix might not be singular and still be hard to invert due to its susceptibility to numerical instability during inversion. So a more general (and useful) concept is the *conditioning* of the matrix - whether a matrix is *well-conditioned* or *ill-conditioned* (close to singular). Again, we can use spectral properties of the matrix to help us define whether a matrix is well-conditioned or not - the *condition number* and the *determinant* of a matrix.

**Condition number** of a matrix $A$ is defined as:

$$\kappa(A) = \Vert A \Vert \Vert A^{-1} \Vert$$

where $\Vert A \Vert$ is the matrix norm of $A$.

For a symetric positive definite matrix $A$ (such as the Gram matrix $X^T X$), the condition number is:

$$\kappa(A) = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}$$

where $\lambda_{\text{max}}$ and $\lambda_{\text{min}}$ are the maximum and minimum eigenvalues of $A$. A *well-conditioned* matrix is the one with low condition number (close to 1), while a *ill-conditioned* matrix is the one with high condition number (typically much larger than 1). A well-conditioned matrix will have all eigenvalues of a similar magnitude, while an ill-conditioned matrix will have a large spread between its eigenvalues and also some of the eigenvalues might be close to zero. These small eigenvalues will correspond to very large eigenvalues of its corresponding inverse matrix. In general, presence of very small eigenvalues causes numerical instability during the inversion of the matrix because small errors in input values and rounding errors during calculations can be amplified in the inverse matrix.

This explains why orthogonal matrices are easy to invert - as all of their eigenvalues are either 1 or -1, their condition number (and the condition number of their inverse) is 1.

**Determinant of a matrix** is a scalar value which measures how much the matrix expands or contracts space. Considering that the Gram matrix $X^T X$ is square and positive semi-definite (all eigenvalues are non-negative) its determinant can be calculated as the product of its eigenvalues:

$$\det(X^T X) = \prod_{i=1}^n \lambda_i$$

where $\lambda_i$ are the eigenvalues of $X^T X$.

The determinant of a matrix is reciprocal to the determinant of its inverse, so this applies to the determinant of our Gram matrix $X^T X$ and its inverse $(X^T X)^{-1}$ as well:

$$\det((X^T X)^{-1}) = \frac{1}{\det(X^T X)}$$

If the determinant of the Gram matrix $X^T X$ is small or close to zero, the determinant of its inverse $(X^T X)^{-1}$ will be very large, which is a sign of ill-conditioning. If the determinant of the Gram matrix $X^T X$ is zero (because at least one of its eigenvalues is zero), the Gram matrix $X^T X$ is singular and its inverse $(X^T X)^{-1}$ does not exist.

In general there is no direct correspondence between matrix's conditioning number and its determinant, but they are indirectly related so far as they both give us insight into the spectral properties of the matrix and its stability during numerical calculations such as inversion. 

Next we will show how to use these spectral properties to derive a new regularized solution to the OLS problem - a ridge regression.

**Ridge regression** is a method for estimating the unknown parameters (coefficients) in a linear regression model using a regularized objective function:

$$L(w) = \Vert X w - y \Vert^2 + \lambda \Vert w \Vert^2$$

where $\lambda$ is the regularization parameter (not to be confused with eigenvalues $\lambda$ which we used earlier!). The ridge regression objective function is a sum of the OLS objective function and a *regularization term* $\lambda \Vert w \Vert^2$ which penalizes large values of the coefficients $w$.

Lets expand the ridge regression objective function:

$$L(w) = (X w - y)^T (X w - y) + \lambda w^T w$$

To find the coefficients $w$ which minimize the ridge regression objective function, we take the gradient of the objective function and set it to zero:

$$\nabla_w L(w) = 2 X^T (X w - y) + 2 \lambda w = 0$$

Solving for $w$:

$$X^T X w + \lambda w = X^T y$$

$$(X^T X + \lambda I) w = X^T y$$

Finally we obtain the ridge regression solution for the coefficients $w$:

$$w = (X^T X + \lambda I)^{-1} X^T y$$

This is the ridge regression solution. The regularization term $\lambda I$ ensures that the Gram matrix $X^T X + \lambda I$ is well-conditioned and invertible, even if the original Gram matrix $X^T X$ is ill-conditioned or singular. The regularization parameter $\lambda$ controls the amount of regularization applied to the coefficients $w$ - the larger the value of $\lambda$, the more the coefficients are shrunk towards zero.

**Regularization - matrix perspective.** In order to gain insight how the regularization term $\lambda I$ affects the spectral properties of the Gram matrix $X^T X$, we can perform an eigenvalue decomposition of the regularized Gram matrix $X^T X + \lambda I$:

$$X^T X + \lambda I = Q \Lambda Q^T + \lambda I = Q \Lambda Q^T + \lambda Q I Q^T = Q (\Lambda + \lambda I) Q^T$$

Note that since $I$ is the identity matrix and $Q$ is an orthogonal matrix we can write $\lambda I = \lambda Q I Q^T$. So the eigenvalues of the regularized Gram matrix $X^T X + \lambda I$ are the original eigenvalues $\lambda_{1}, \dots, \lambda_{n}$ of the Gram matrix $X^T X$ increased by $\lambda$, so $\lambda_{1} + \lambda, \dots, \lambda_{n} + \lambda$. This has several benefits for the conditioning of the Gram matrix $X^T X$:

1. *Non-singularity*: The regularization term $\lambda I$ ensures that the regularized Gram matrix $X^T X + \lambda I$ has all non-zero eigenvalues, even if the original Gram matrix $X^T X$ has some zero eigenvalues. This ensures that the regularized Gram matrix is full rank and non-singular.

2. *Better conditioning*: The regularization term $\lambda I$ increases the eigenvalues of the Gram matrix $X^T X$ by $\lambda$, which decreases the difference between the largest and smallest eigenvalues, which in turn decreases the condition number of the regularized Gram matrix $X^T X + \lambda I$ and makes it better conditioned.

**Intuition behind regularization.** We can argue that adding the regularization term $\lambda I$ does not fundamentally change the correlation matrix $X^T X$ - the matrix still square, symmetrical, positive semi-definite (all eigenvalues are still non-negative), and encodes the same basic correlation information between features of our data matrix $X$ (correlations are encoded in off-diagonal elements of $X^T X$ which remain unchanged). However, by increasing all the eigenvalues by $\lambda$ the matrix $X^T X$ effectively becomes more orthogonal, meaning that features are less correlated and the matrix is better conditioned and more stable during inversion.














