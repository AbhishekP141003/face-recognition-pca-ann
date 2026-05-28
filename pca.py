"""
pca.py
------
Full manual PCA (Eigenfaces) implementation.
Follows every mathematical step from the project spec exactly.

Steps implemented:
  STEP 2  - Mean Calculation          M in R^(mn x 1)
  STEP 3  - Mean Centering            Delta = Face_Db - M
  STEP 4  - Surrogate Covariance      C(p,p) instead of C(mn,mn)
  STEP 5  - Eigenvalue Decomposition  using SciPy eigh
  STEP 6  - Select Best k Directions  Psi in R^(p x k)
  STEP 7  - Generate Eigenfaces       Phi in R^(k x mn)
  STEP 8  - Generate Signatures       omega in R^(k x p)
"""

import numpy as np
from scipy.linalg import eigh    # SciPy for eigen decomposition (allowed)


class PCA:
    """
    Manual PCA for face recognition (Eigenfaces method).

    Attributes
    ----------
    k            : int       number of eigenfaces to keep
    M            : ndarray   mean face vector  (mn,)
    eigenfaces   : ndarray   Phi matrix        (k, mn)
    eigenvalues  : ndarray   top-k eigenvalues (k,)
    explained_var: ndarray   variance ratio    (k,)
    """

    def __init__(self, n_components):
        """
        Parameters
        ----------
        n_components : int
            Number of top eigenfaces (k) to retain.
        """
        self.k             = n_components
        self.M             = None     # mean face
        self.eigenfaces    = None     # Phi: (k, mn)
        self.eigenvalues_  = None
        self.explained_var = None

    # =========================================================================
    # STEP 2 — Mean Face Calculation
    #
    # Formula:  M = (1/p) * sum over all p images
    # M in R^(mn x 1)
    #
    # Why? The mean face represents the "average" face.
    # We subtract it to centre the data around zero.
    # =========================================================================

    def _compute_mean(self, Face_Db):
        """
        Compute mean face M from Face_Db.

        Parameters
        ----------
        Face_Db : ndarray (mn, p)

        Returns
        -------
        M : ndarray (mn,)   — mean face vector
        """
        # Average across all p images (axis=1 means across columns)
        M = np.mean(Face_Db, axis=1)   # shape: (mn,)
        print(f"\n[STEP 2] Mean face computed   M.shape = {M.shape}")
        return M

    # =========================================================================
    # STEP 3 — Mean Centering (Zero-Mean)
    #
    # Formula:  Delta(i) = Face_Db(i) - M   for i = 1..p
    # Delta in R^(mn x p)
    #
    # Why? Centering removes the effect of brightness/average intensity.
    # PCA finds directions of maximum VARIANCE from the mean, not from zero.
    # Without centering, the first component would just capture the mean.
    # =========================================================================

    def _mean_center(self, Face_Db, M):
        """
        Subtract mean face from every image column.

        Parameters
        ----------
        Face_Db : ndarray (mn, p)
        M       : ndarray (mn,)

        Returns
        -------
        Delta : ndarray (mn, p)   — mean-centred face matrix
        """
        # Broadcasting: M.reshape(-1,1) makes M a column vector (mn,1)
        # Subtracted from every column of Face_Db automatically
        Delta = Face_Db - M.reshape(-1, 1)    # (mn, p)
        print(f"[STEP 3] Mean centering done  Delta.shape = {Delta.shape}")
        return Delta

    # =========================================================================
    # STEP 4 — Surrogate (Compact) Covariance Matrix
    #
    # Standard covariance: C = Delta * Delta^T   shape: (mn, mn)
    # e.g. 4096 x 4096 = 16 MILLION elements — too expensive!
    #
    # Turk & Pentland trick (surrogate):
    #   C_surrogate = Delta^T * Delta   shape: (p, p)
    # e.g. 360 x 360 = much smaller!
    #
    # Key insight:
    #   If v is eigenvector of Delta^T*Delta with eigenvalue λ, then
    #   Delta*v is eigenvector of Delta*Delta^T with the same eigenvalue λ.
    #
    # So we compute eigenvectors in (p,p) space and convert to (mn,mn) space.
    # =========================================================================

    def _surrogate_covariance(self, Delta):
        """
        Compute surrogate covariance matrix C = Delta^T * Delta.

        Parameters
        ----------
        Delta : ndarray (mn, p)

        Returns
        -------
        C : ndarray (p, p)   — surrogate covariance
        """
        mn, p = Delta.shape
        # C = Delta^T * Delta   shape (p, p)
        C = (Delta.T @ Delta) / (p - 1)
        print(f"[STEP 4] Surrogate covariance  C.shape = {C.shape}  "
              f"(instead of {mn}x{mn})")
        return C

    # =========================================================================
    # STEP 5 — Eigenvalue and Eigenvector Decomposition
    #
    # Solve: C * v = lambda * v
    #
    # scipy.linalg.eigh returns eigenvalues in ascending order.
    # We reverse to get DESCENDING order (largest first).
    #
    # Then convert small-space eigenvectors (p,) back to full space (mn,):
    #   u_i = Delta * v_i    (then normalise)
    # =========================================================================

    def _eigen_decompose(self, C, Delta):
        """
        Eigen-decompose surrogate covariance C.
        Convert eigenvectors back to original (mn) space.

        Parameters
        ----------
        C     : ndarray (p, p)   — surrogate covariance
        Delta : ndarray (mn, p)  — mean-centred faces

        Returns
        -------
        eigenvalues  : ndarray (p,)    — sorted descending
        eigenvectors : ndarray (mn, p) — full-space eigenvectors (columns)
        """
        # eigh: returns (eigenvalues, eigenvectors) in ASCENDING order
        # Only valid for symmetric matrices — covariance is always symmetric
        eigenvalues, V = eigh(C)        # V shape: (p, p), columns = eigenvectors

        # Convert from surrogate space (p) back to image space (mn)
        # Formula: u_i = Delta * v_i
        U = Delta @ V                   # shape: (mn, p)

        # Normalise each column vector to unit length
        norms = np.linalg.norm(U, axis=0, keepdims=True)  # (1, p)
        norms[norms == 0] = 1
        U = U / norms                   # (mn, p) unit eigenvectors

        # Sort in DESCENDING order of eigenvalue
        idx         = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U           = U[:, idx]

        print(f"[STEP 5] Eigen decomposition done")
        print(f"         eigenvalues.shape  = {eigenvalues.shape}")
        print(f"         eigenvectors.shape = {U.shape}  (mn x p)")
        return eigenvalues, U

    # =========================================================================
    # STEP 6 — Select Best k Directions
    #
    # Feature vector matrix Psi in R^(p x k)
    # Keep only the top-k eigenvectors (columns of U).
    #
    # Why k matters:
    #   - Small k: fast but less discriminative (may lose facial detail)
    #   - Large k: more detail but may overfit / include noise directions
    # =========================================================================

    def _select_top_k(self, eigenvalues, U):
        """
        Select top-k eigenvectors and eigenvalues.

        Parameters
        ----------
        eigenvalues : ndarray (p,)
        U           : ndarray (mn, p)

        Returns
        -------
        top_k_vals : ndarray (k,)
        top_k_vecs : ndarray (mn, k)   — columns are the k eigenvectors
        """
        k = min(self.k, U.shape[1])
        top_k_vals = eigenvalues[:k]    # (k,)
        top_k_vecs = U[:, :k]           # (mn, k)

        explained  = top_k_vals / (eigenvalues.sum() + 1e-12)
        self.explained_var = explained

        print(f"[STEP 6] Selected k = {k} directions  (Psi shape: {top_k_vecs.shape})")
        print(f"         Variance explained by top-{k}: "
              f"{explained.sum()*100:.1f}%")
        return top_k_vals, top_k_vecs

    # =========================================================================
    # STEP 7 — Generate Eigenfaces
    #
    # Formula: Phi(k x mn) = Psi^T * Delta^T
    #
    # Psi   : top-k eigenvectors in original space — shape (mn, k)
    # Phi   : eigenfaces matrix                    — shape (k, mn)
    #
    # Each ROW of Phi is one eigenface (can be reshaped to an image).
    # =========================================================================

    def _generate_eigenfaces(self, top_k_vecs):
        """
        Generate eigenfaces matrix Phi.

        Parameters
        ----------
        top_k_vecs : ndarray (mn, k)

        Returns
        -------
        Phi : ndarray (k, mn)   — each row = one eigenface
        """
        Phi = top_k_vecs.T     # (k, mn)
        print(f"[STEP 7] Eigenfaces generated  Phi.shape = {Phi.shape}  (k x mn)")
        return Phi

    # =========================================================================
    # STEP 8 — Generate Face Signatures
    #
    # Formula: omega(k x p) = Phi(k x mn) * Delta(mn x p)
    #
    # Each COLUMN of omega is the "signature" (feature vector) of one image.
    # These are low-dimensional representations used to train the ANN.
    # =========================================================================

    def _generate_signatures(self, Phi, Delta):
        """
        Project all faces onto eigenface space to get signatures.

        Parameters
        ----------
        Phi   : ndarray (k, mn)
        Delta : ndarray (mn, p)

        Returns
        -------
        omega : ndarray (k, p)   — each column = signature of one image
        """
        omega = Phi @ Delta     # (k, p)
        print(f"[STEP 8] Signatures generated  omega.shape = {omega.shape}  (k x p)")
        return omega

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def fit(self, Face_Db):
        """
        Run all PCA steps on training Face_Db.

        Parameters
        ----------
        Face_Db : ndarray (mn, p)   — training face database

        Returns
        -------
        self
        """
        print(f"\n--- PCA FIT  (k={self.k}) ---")

        # Step 2: mean face
        self.M = self._compute_mean(Face_Db)

        # Step 3: mean centering
        Delta  = self._mean_center(Face_Db, self.M)

        # Step 4: surrogate covariance
        C      = self._surrogate_covariance(Delta)

        # Step 5: eigen decomposition
        eigenvalues, U = self._eigen_decompose(C, Delta)

        # Step 6: select top-k
        top_k_vals, top_k_vecs = self._select_top_k(eigenvalues, U)
        self.eigenvalues_ = top_k_vals

        # Step 7: eigenfaces
        self.eigenfaces = self._generate_eigenfaces(top_k_vecs)   # (k, mn)

        # Step 8: training signatures
        self._Delta_train = Delta
        self.signatures_train = self._generate_signatures(
            self.eigenfaces, Delta)                                # (k, p)

        return self

    def transform(self, Face_Db_new):
        """
        Project new images (mean-centred) onto eigenface space.

        Testing formula:
          I2    = I1 - M          (mean-centred test image)
          Omega = Phi * I2        (project to eigenface space)

        Parameters
        ----------
        Face_Db_new : ndarray (mn, n)   — new images (columns)

        Returns
        -------
        omega_new : ndarray (k, n)   — signatures of new images
        """
        # Subtract training mean from test images
        Delta_new = Face_Db_new - self.M.reshape(-1, 1)   # (mn, n)
        omega_new = self.eigenfaces @ Delta_new            # (k, n)
        return omega_new

    def fit_transform(self, Face_Db):
        """Fit PCA and return training signatures."""
        self.fit(Face_Db)
        return self.signatures_train