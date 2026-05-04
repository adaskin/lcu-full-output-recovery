import numpy as np
from numpy.linalg import norm
import pennylane as qml
import matplotlib.pyplot as plt
from quantum_circuit import AlternativeLCU




# ======================= tools =========================================
def random_unitary(dim, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(Z)
    return Q

def random_permutation(dim, seed=None):
    rng = np.random.default_rng(seed)
    return np.eye(dim)[rng.permutation(dim)]



# ================== Standard LCU ======================================
def standard_lcu_prob(K, alphas, U_list, psi):
    """
    Standard LCU block‑encoding (Childs–Wiebe).
    ancilla size = ceil(log2(K)), system size = log2(dim).
    Coefficients assumed real and non‑negative.
    Returns success probability (ancilla measured in |0...0>).
    """
    dim = len(psi)
    n_sys = int(np.log2(dim))
    n_idx = int(np.ceil(np.log2(K)))
    n_wires = n_idx + n_sys          # only ancilla + system, no rotation qubit
    dev = qml.device("default.qubit", wires=n_wires, shots=None)

    # Build normalised amplitude vector for ancilla
    ampl = np.zeros(2**n_idx)
    ampl[:K] = np.sqrt(alphas)      # alphas are positive
    ampl = ampl / np.linalg.norm(ampl)   # norm = 1

    @qml.qnode(dev)
    def circuit():
        # Prepare ancilla state
        qml.StatePrep(ampl, wires=range(n_idx))
        # Load system state
        qml.StatePrep(psi, wires=range(n_idx, n_wires))

        # Controlled unitaries
        for t in range(K):
            U = U_list[t]
       
            binary = f"{t:0{n_idx}b}"
            ctrl_vals = [int(b) for b in binary]
            qml.ctrl(
                qml.QubitUnitary(U, wires=range(n_idx, n_wires)),
                control=range(n_idx),
                control_values=ctrl_vals
            )

        # Uncompute ancilla (adjoint of preparation)
        qml.adjoint(qml.StatePrep)(ampl, wires=range(n_idx))

        # Measure ancilla
        return qml.probs(wires=range(n_idx))

  
    probs = circuit()  # provide psi to the circuit
    return probs[0]   # all ancillas in |0>

# ===================== Experiment ======================================

  
# if __name__ == '__main__':

K=4
n_sys = 4
n_trials=10
dim = 2**n_sys
low_vals = np.linspace(0.1, 1, 5)  # Vary the "low" coefficient from 0.4 to 1.0
rng = np.random.default_rng(42)

results = {'alternative_with_rot': [],
            'alternative_without_rot': [], 
            'std_lcu': [], 'std_lcu_analytic': [],
            'alternative_without_rot_analytic': [], 
            'alternative_with_rot_analytic': []}

for low in low_vals:
    alphas = np.zeros(K)
    alphas[:K//2] = 1         # half of the terms have coefficient 1
    alphas[K//2:] = low         # other half have coefficient 'low'

    p_std, p_std_an = [], []
    p_with_rot_an, p_without_rot_an = [], []
    p_without_rot, p_with_rot = [], []

    for _ in range(n_trials):
        U_list = [random_unitary(dim, seed=rng.integers(0, 1e7)) for _ in range(K)]
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)

        # --- Simulation using AlternativeLCU ---
        # 1) Success with only index qubits (no rotation qubit in measurement)
        lcu_idx = AlternativeLCU(K, alphas, U_list, psi,
                                    single_channel=True,      # rotation qubit stays |0>
                                    output="probs_ancilla",
                                    include_rot_in_ancilla_probs=False)
        probs = lcu_idx.run(psi=psi)          # probability distribution over index wires
        p_without_rot.append(probs[0])      # all index qubits = 0

        # 2) Success with index + rotation qubits all zero
        lcu_all = AlternativeLCU(K, alphas, U_list, psi,
                                    single_channel=True,
                                    output="probs_ancilla",
                                    include_rot_in_ancilla_probs=True)
        probs_all = lcu_all.run(psi=psi)
        p_with_rot.append(probs_all[0])    # all index and rotation qubits = 0



        p_std.append(standard_lcu_prob(K, alphas, U_list, psi))

        # Analytic standard LCU probability
        T = sum(alphas[t] * U_list[t] for t in range(K))
        Tpsi = T @ psi
        nTpsi = np.linalg.norm(Tpsi)

        T2 = sum(np.sqrt(1-alphas[t]**2) * U_list[t] for t in range(K))
        T2psi = T2 @ psi
        n2T2psi = np.linalg.norm(T2psi)
        s = np.sum(alphas)
        # a_max = 1#np.max(np.abs(alphas))
        p_std_an.append(nTpsi**2 / s**2)
        p_with_rot_an.append(nTpsi**2 / (K**2))  # Placeholder for new method analytic
        p = (nTpsi**2 + (n2T2psi)**2) / (K**2) 
        p_without_rot_an.append(p)

    results['alternative_without_rot'].append(np.mean(p_without_rot))
    results['alternative_with_rot'].append(np.mean(p_with_rot))
    results['std_lcu'].append(np.mean(p_std))
    results['std_lcu_analytic'].append(np.mean(p_std_an))

    results['alternative_without_rot_analytic'].append(np.mean(p_without_rot_an))
    results['alternative_with_rot_analytic'].append(np.mean(p_with_rot_an))

# Plot
plt.rcParams.update({
'figure.figsize': (7, 5), 'font.size': 14, 'axes.labelsize': 14,
'axes.titlesize': 14, 'legend.fontsize': 12, 'xtick.labelsize': 14,
'ytick.labelsize': 14, 'lines.linewidth': 2, 
'savefig.dpi': 300, 'savefig.bbox': 'tight', 'text.usetex': False,
})
plt.figure()
plt.plot(low_vals, results['alternative_without_rot'], 'o-', label='Without rotation qubit')
plt.plot(low_vals, results['alternative_without_rot_analytic'], 'd--', label='Without rotation qubit (analytic)')
plt.plot(low_vals, results['alternative_with_rot'], 's-', label='With rotation qubit')
plt.plot(low_vals, results['alternative_with_rot_analytic'], 'v--', label='With rotation qubit (analytic)')
plt.plot(low_vals, results['std_lcu'], 's-', label='Standard LCU')
plt.plot(low_vals, results['std_lcu_analytic'], 'x--', label='Standard LCU (analytic)')


plt.plot(low_vals, [1/n_sys]*len(low_vals), 'b-', label='Polynomial line $1/n$')
plt.plot(low_vals, [1/2**n_sys]*len(low_vals), 'r-', label='Exponential line $1/2^n$')

plt.xlabel('Low coefficient value (half of the terms)')
plt.ylim(0,0.5 )  # Adjust y-axis limit for better visibility
plt.ylabel('Success probability $p_{\\text{succ}}$')
plt.title(f'Probability Comparison with $K={K}$, $N={2**n_sys}$')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('comparison_std_vs_alternative.pdf')
plt.savefig('comparison_std_vs_alternative.png')
plt.show()
print("Results:", results)