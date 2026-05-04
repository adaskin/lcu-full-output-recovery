import numpy as np
import pennylane as qml
from scipy.linalg import block_diag, hadamard
from numpy.linalg import norm

# ================== AlternativeLCU class ================================
class AlternativeLCU:
    """
    Block‑encoding circuit for the LCU (linear combination of unitaries) and
    theoretical full output matrix for verification.

    Parameters
    ----------
    K : int
        Number of unitaries.
    alpha : array_like (K,)
        Coefficients (real, positive, <= 1).
    U_list : list of ndarray (N,N)
        List of unitary matrices.
    psi : array_like (N,)
        Input state vector.
    single_channel : bool
        If True, use Hadamard on rotation qubit to prepare |+>, else keep |0>.
    output : str
        Type of circuit output: "samples", "probs_ancilla", "probs_full", "full_state".
    include_rot_in_ancilla_probs : bool
        Only relevant for output="probs_ancilla". If True, include rotation qubit.
    """
    def __init__(self, K, alpha, U_list, psi=None,
                 single_channel=True,
                 output="samples",sample_shots=1,
                 include_rot_in_ancilla_probs=True,
                 compute_outputs_via_matrices=False):
        self.K = K
        self.alpha = np.asarray(alpha)
        self.U_list = U_list
        self.psi = np.asarray(psi)
        self.dim = len(U_list[0])  # assuming square unitaries
        self.n_sys = int(np.log2(self.dim))
        self.N = self.dim
        self.n_idx = int(np.ceil(np.log2(K)))
        self.n_wires = self.n_idx + 1 + self.n_sys   # index, rotation, system
        self.single_channel = single_channel
        self.output = output
        self.include_rot_in_ancilla_probs = include_rot_in_ancilla_probs
        if compute_outputs_via_matrices:
            if self.psi is None:
                raise ValueError("Input state 'psi' must be provided to compute outputs via matrices.")
            self.compute_outputs_via_matrices()
        # Device and QNode (built
        # if output == "samples":
        self.sample_shots = sample_shots
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        self.circuit = self._build_circuit



    @staticmethod
    def R_reflection_with_w(w):
        """Reflection variant: [[w, r], [r, -w]]. Unitary when w² + r² = 1."""
        r = np.sqrt(1 - w**2)
        R = np.array([[w, r], [r, -w]], dtype=complex)
        assert np.allclose(R.conj().T @ R, np.eye(2)), f"R not unitary for w={w}"
        return R
    @staticmethod
    def R_reflection_with_r(r):
        """Reflection variant: [[w, r], [r, -w]]. Unitary when w² + r² = 1."""
        w = np.sqrt(1 - r**2)
        R = np.array([[w, r], [r, -w]], dtype=complex)
        assert np.isclose(np.abs(R[0,0])**2 + np.abs(R[0,1])**2, 1.0), "Coefficients do not satisfy"
        return R
    

    def _build_circuit(self, psi=None):
        def circuit_gates(psi=None):
            # Prepare system state
            if psi is None:
                if self.psi is None:
                    raise ValueError("Input state 'psi' must be provided to the circuit.")
                Warning("No input state provided to circuit, using instance's psi.")
                psi = self.psi
            psi = psi / norm(psi)  # ensure normalized
            qml.StatePrep(psi, wires=range(self.n_idx + 1, self.n_wires))
            
            # Index superposition
            for i in range(self.n_idx):
                qml.Hadamard(wires=i)
            # Optional Hadamard on rotation qubit
            if not self.single_channel:
                qml.Hadamard(wires=self.n_idx)

            # Controlled block encoding
            for t in range(self.K):
                R = self.R_reflection_with_w(self.alpha[t])
                U = self.U_list[t]
                gate_matrix = np.kron(R, U)
                binary = f"{t:0{self.n_idx}b}"
                ctrl_vals = [int(b) for b in binary]
                qml.ctrl(
                    qml.QubitUnitary(gate_matrix,
                                     wires=[self.n_idx] + list(range(self.n_idx + 1, self.n_wires))),
                    control=range(self.n_idx),
                    control_values=ctrl_vals
                )

            # Final index Hadamards
            for i in range(self.n_idx):
                qml.Hadamard(wires=i)

            #SWAP rotation qubit to the left of index qubits for easier measurement
            if self.n_idx == 1:
                qml.SWAP(wires=[0,1])  # Swap idx and rot for single index case
            else:
                rwire = self.n_idx          # rotation starts at n_idx
                for i in range(self.n_idx): # move it left n_idx times
                    qml.SWAP(wires=[rwire, rwire - 1])
                    rwire -= 1

        @qml.qnode(self.dev)
        def qnode_with_probs(psi=None):
 
            circuit_gates(psi=psi)
            
            # Output selection
            if self.output == "probs_ancilla":
                # return qml.state()  # Return full state for debugging
                if self.include_rot_in_ancilla_probs:
                    return qml.probs(wires=range(0, self.n_idx+1))
                else:
                    return qml.probs(wires=range(1, self.n_idx+1))
            elif self.output == "probs_full":
                return qml.probs(wires=range(self.n_wires))
            elif self.output == "full_state":
                return qml.state()
            
        @qml.set_shots(shots=self.sample_shots)
        @qml.qnode(self.dev)
        def qnode_with_samples(psi=None):

            circuit_gates(psi=psi)
            
            if self.output == "samples":
                return qml.sample(wires=range(self.n_wires))
            else:
                raise ValueError(f"Unknown output type: {self.output}")
        if self.output == "samples":
            return qnode_with_samples(psi=psi)
        else:
            return qnode_with_probs(psi=psi)

    def run(self, psi=None):
        """Execute the quantum circuit with the specified number of shots."""
        if psi is None and self.psi is None:
            raise ValueError("Input state 'psi' is not provided.")
        elif psi is None and self.psi is not None:
            psi = self.psi # Use the instance's psi if not provided
        return self.circuit(psi=psi)
    
    def compute_outputs_via_matrices(self):
        """Compute the exact full output matrix Phi (2K x N) and target vector phi."""
        w = self.alpha
        r = np.sqrt(1 - w**2)
        H = hadamard(self.K) / np.sqrt(self.K)
        I2N = np.eye(2 * self.N)

        # Block-diagonal M
        M_blocks = []
        for k in range(self.K):
            Rk = np.array([[w[k], r[k]], [r[k], -w[k]]])
            M_blocks.append(np.kron(Rk, self.U_list[k]))
        M_mat_diag = block_diag(*M_blocks)   # (2*N*K) x (2*N*K)

        # Unshuffled unitary: (H ⊗ I) M (H^T ⊗ I)
        # U_unshuffled = np.kron(H, I2N) @ M_mat_diag @ np.kron(H.T, I2N)

        # Shuffle permutation: group by (rot, idx) then system
        half = self.N * self.K
        perm = []
        for rot in range(2):
            for idx in range(self.K):
                start = idx * (2 * self.N) + rot * self.N
                perm.extend(range(start, start + self.N))
        Shuffle = np.eye(2 * half)[perm]
        
        #WE SHOULD VECTORIZE THIS
        # U_shuffled = Shuffle @ U_unshuffled @ Shuffle.T

        # Input state: idx=0, rot=0, system=psi
        input_vec = np.zeros(self.K * 2 * self.N, dtype=complex)
        input_vec[:self.N] = self.psi
        # U_unshuffled = np.kron(H, I2N) @ M_mat_diag @ np.kron(H.T, I2N)
        # full output: (index)⊗(rot) ⊗ (sys).
        # self.out_unshuffled = U_unshuffled @ input_vec
        self.out_unshuffled =  np.kron(H, I2N) @ (M_mat_diag @ (np.kron(H.T, I2N) @ input_vec))
        # Full output(shuffled basis):(rot) ⊗ (index) ⊗ (sys).
        self.out_shuffled = Shuffle @ self.out_unshuffled
   
        # Reshape to (2K, N) matrix
        self.Phi_full_true2KN = self.out_shuffled.reshape(2 * self.K, self.N)
           
        # Reshape to (K, 2N) matrix
        self.Phi_full_trueK2N = self.out_unshuffled.reshape(self.K, 2*self.N)
        self.phi_target_true = self.Phi_full_true2KN[0, :]  # ancilla=0 block gives T|psi>

    @property
    def true_output_matrix2KN(self):
        """Return the theoretically exact full output matrix (2K x N)."""
        return self.Phi_full_true2KN
    @property
    def true_output_matrixK2N(self):
        """Return the theoretically exact full output matrix (2K x N)."""
        return self.Phi_full_trueK2N
    @property
    def true_target_vector(self):
        """Return the theoretically exact target vector phi = T|psi>."""
        return self.phi_target_true

    @property
    def mixing_matrix(self):
        """The K x K mixing matrix that relates the rot=0 block to the unitaries."""
        return self.M_mat




def R_reflection_with_r(r):
    """Reflection variant: [[w, r], [r, -w]]. Unitary when w² + r² = 1."""
    w = np.sqrt(1 - r**2)
    R = np.array([[w, r], [r, -w]], dtype=complex)
    assert np.isclose(np.abs(R[0,0])**2 + np.abs(R[0,1])**2, 1.0), "Coefficients do not satisfy"
    return R

def R_reflection_with_w(w):
    """Reflection variant: [[w, r], [r, -w]]. Unitary when w² + r² = 1."""
    r = np.sqrt(1 - w**2)
    R = np.array([[w, r], [r, -w]], dtype=complex)
    assert np.isclose(np.abs(R[0,0])**2 + np.abs(R[0,1])**2, 1.0), "Coefficients do not satisfy"


    return R

def R_reflection_from_alpha(alpha):
    """
    Compute the reflection rotation gate [[w, r], [r, -w]] such that
    w + r = alpha (when using |+⟩ rotation initialization).
    
    Requires: |alpha| <= sqrt(2)
    """
    alpha = np.real(alpha)  # assuming real coefficients for now
    assert abs(alpha) <= np.sqrt(2), f"alpha={alpha} exceeds sqrt(2)"
    
    # Take the larger root for numerical stability
    w = (alpha + np.sqrt(2 - alpha**2)) / 2
    r = np.sqrt(1 - w**2)
    
    # Verify w + r = alpha
    assert np.isclose(w + r, alpha), f"Failed: w+r={w+r}, alpha={alpha}"
    R = np.array([[w, r], [r, -w]], dtype=complex)
    assert np.allclose(R.conj().T @ R, np.eye(2)), f"R is not unitary for alpha={alpha}, w={w}, r={r}"
    return R    

