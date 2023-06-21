import warnings
class ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel):
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * ||w||_1 + 0.5 * b * ||w||_2^2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter. ``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.

    max_iter : int, default=1000
        The maximum number of iterations.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``, see Notes below.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).

    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_targets, n_features)
        Sparse representation of the `coef_`.

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : list of int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : float or ndarray of shape (n_targets,)
        Given param alpha, the dual gaps at the end of the optimization,
        same shape as each observation of y.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ElasticNetCV : Elastic net model with best model selection by
        cross-validation.
    SGDRegressor : Implements elastic net regression with incremental training.
    SGDClassifier : Implements logistic regression with elastic net penalty
        (``SGDClassifier(loss="log_loss", penalty="elasticnet")``).

    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    The precise stopping criteria based on `tol` are the following: First, check that
    that maximum coordinate update, i.e. :math:`\\max_j |w_j^{new} - w_j^{old}|`
    is smaller than `tol` times the maximum absolute coefficient, :math:`\\max_j |w_j|`.
    If so, then additionally check whether the dual gap is smaller than `tol` times
    :math:`||y||_2^2 / n_{\text{samples}}`.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> regr = ElasticNet(random_state=0)
    >>> regr.fit(X, y)
    ElasticNet(random_state=0)
    >>> print(regr.coef_)
    [18.83816048 64.55968825]
    >>> print(regr.intercept_)
    1.451...
    >>> print(regr.predict([[0, 0]]))
    [1.451...]
    """
    """
    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "fit_intercept": ["boolean"],
        "precompute": ["boolean", "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "copy_X": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "warm_start": ["boolean"],
        "positive": ["boolean"],
        "random_state": ["random_state"],
        "selection": [StrOptions({"cyclic", "random"})],
    }
    """

    path = staticmethod(enet_path)

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weights. Internally, the `sample_weight` vector will be
            rescaled to sum to `n_samples`.

            .. versionadded:: 0.23

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        device = X.device
        if self.alpha == 0:
            warnings.warn(
                (
                    "With alpha=0, this algorithm does not converge "
                    "well. You are advised to use the LinearRegression "
                    "estimator"
                ),
                stacklevel=2,
            )

        # Remember if X is copied
        X_copied = False
        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        n_samples, n_features = X.shape
        alpha = self.alpha

        # Ensure copying happens only once, don't do it again if done above.
        # X and y will be rescaled if sample_weight is not None, order='F'
        # ensures that the returned X and y are still F-contiguous.

        X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
            X,
            y,
            None,
            self.precompute,
            normalize=False,
            fit_intercept=self.fit_intercept,
            copy=False,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # coordinate descent needs F-ordered arrays and _pre_fit might have
        # called _rescale_data
        if check_input or sample_weight is not None:
            X, y = _set_order(X, y, order="F")
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_targets = y.shape[1]

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = torch.zeros((n_targets, n_features), dtype=X.dtype, order="F")
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []

        for k in range(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            _, this_coef, this_dual_gap, this_iter = self.path(
                X,
                y[:, k],
                l1_ratio=self.l1_ratio,
                eps=None,
                n_alphas=None,
                alphas=[alpha],
                precompute=precompute,
                Xy=this_Xy,
                copy_X=True,
                coef_init=coef_[k],
                verbose=False,
                return_n_iter=True,
                positive=self.positive,
                check_input=False,
                # from here on **params
                tol=self.tol,
                X_offset=X_offset,
                X_scale=X_scale,
                max_iter=self.max_iter,
                random_state=self.random_state,
                selection=self.selection,
                sample_weight=sample_weight,
            )
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]
            self.coef_ = coef_[0]
            self.dual_gap_ = dual_gaps_[0]
        else:
            self.coef_ = coef_
            self.dual_gap_ = dual_gaps_

        self._set_intercept(X_offset, y_offset, X_scale)

        # check for finiteness of coefficients
        if not all(torch.isfinite(w).all() for w in [self.coef_, self.intercept_]):
            raise ValueError(
                "Coordinate descent iterations resulted in non-finite parameter"
                " values. The input data may contain large values and need to"
                " be preprocessed."
            )

        # return self for chaining fit and predict calls
        return self


