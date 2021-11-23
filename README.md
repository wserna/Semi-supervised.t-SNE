# SS.t-SNE
## Semi-supervised neighbor embedding with multi-scale neighborhood preservation
<hr />
This project introduces a semi-supervised version of t-SNE for dimensionality reduction based in neighborhood preservation. The neighborhood size for labeled samples is chosen in such a way that data of the same class predominate (cat-SNE), ignoring in a smart way the unlabeled samples without affecting the normalization factor of the distribution function. The neighborhood size for unlabeled samples follow a multi-scale perspective to preserve local and global structure on the entire dataset.

FUNCTION

Y = SStSNE(X,L,dim,thresh_cat,itr,tol,fpt)

Principal component are used to initialize embedding Y. The optimization involves a line search with backtracking for the step size adjustment under the strong Wolfe conditions. The search direction is the product of the gradient with a diagonal estimate of the Hessian, which is refined by the limited-memory BFGS algorithm (m=7).
The optimization is multiscale, in the sense that BFGS is run several times, starting with the largest perplexity value and then introducing or switching to smaller ones.

Inputs:
  + X  : n x m matrix where each row is a point in HD space with m variables
  + L  : n x 1 matrix where L\(i\) is the class of sample X\(i,:\)
       if L\(i\) == -1, X\(i,:\)is an unlabeled point.
  + dim : embedding dimensionality \(scalar integer\)
  + thresh_cat : \(\[0.5,1\[\) Percentage of points of the same class in the soft neighborhood of a labeled point.
  + itr : maximum number of iterations per perplexity value \(scalar integer\)
  + tol : tolerance for stopping criterion \(tiny scalar float; 1e-4\)
  + mso : true or false for Multi-scale optimization.
  + fpt : specifies the floating-point type

Outputs:
  + Y   : coordinates in the LD space of the final embedding

References:

+ [1] John A. Lee
    Type 1 and 2 mixtures of divergences for stochastic neighbor embedding
    Proc. ESANN 2012, 20:525-530.
+ [2] J. A. Lee, E. Renard, G. Bernard, P. Dupont, M. Verleysen
    Type 1 and 2 mixtures of Kullback-Leibler divergences
    as cost functions in dimensionality reduction
    based on similarity preservation
    Neurocomputing 2013, 112: 92-108.
+ [3] John A. Lee, Diego H. Peluffo, Michel Verleysen
    Multi-scale similarities in stochastic neighbour embedding:
    Reducing dimensionality while preserving both local and global structure
    Neurocomputing 2015, 169:246-261.
    http://dx.doi.org/10.1016/j.neucom.2014.12.095
+ [4] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & 
    Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
