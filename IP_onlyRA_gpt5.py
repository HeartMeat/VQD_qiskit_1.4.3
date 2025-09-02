"""
H2 (STO-3G) 전자구조의 모든 에너지 준위를 Variational Quantum Deflation(VQD)로 추정하는 스크립트

핵심 구현/참고(논문: Higgott, Wang, Brierley, Quantum 2019, q-2019-07-01-156):
  - 목적함수(식 (2)):  F(λ_k) = <H>_{λ_k} + Σ_{i<k} β_i |<ψ(λ_k)|ψ(λ_i)>|^2
    · §2 “Variational quantum deflation algorithm” / Eq. (2)
  - 유효 해밀토니안(식 (4)): H_k = H + Σ_{i<k} β_i |i><i|
    · §3 “Overlap weighting” / Eq. (4)
  - 저깊이 오버랩(§4): |<ψ(λ_i)|ψ(λ_k)>|^2 = |<0| R(λ_i)^† R(λ_k) |0>|^2
    · §4 “Low-depth implementations”
  - 샘플링 상한(식 (3)): Appendix A 상한식을 본문 §2 마지막 단락에도 재기술
    · Eq. (3)

본 스크립트의 설계 선택:
  1) Ansatz: Hartree–Fock 초기상태 + RealAmplitudes (VQD 핵심만 재현하려는 편의적 선택)
     - 논문은 일반화 UCCGSD를 사용하지만, 여기서는 회로 깊이/구현 편의를 위해 RealAmplitudes를 사용.
     - 단, 목적함수/디플레이션/오버랩 회로는 논문과 동일 수학식을 구현.

  2) 오버랩 추정: 저깊이 회로 R(λ_i)^† R(λ_k) |0> 준비 후, |0…0> 확률 평가
     - 실제 하드웨어에서는 반복 샷으로 빈도 추정(§4), 본 코드는 이상적 시뮬레이터(Statevector)로
       정확도를 얻어 알고리즘 구조를 명확히 확인.

  3) (★수정 1) 핵-반발 에너지 E_nuc 처리:
     - Qiskit Nature는 전자 해밀토니안에 핵-반발을 포함하지 않으며, 총 에너지 계산 시 상수로 더함이 표준.
       (공식 튜토리얼/노트북에서 명시)  —> 본 스크립트는 연산자 수준에서 안전하게 I 항으로 시프트.
         H = H_elec + E_nuc * I   (SparsePauliOp)
     - 이렇게 하면 내부 ElectronicEnergy 객체를 변형하지 않고, 버전 의존성이 낮아짐.
     - 참고: Qiskit Nature 튜토리얼/문서에서는 “핵-반발은 후처리 상수” 흐름을 명시.
       (여기서는 연산자 합산으로 ‘동일 효과’를 얻는 안전한 구현)
"""

import numpy as np
from scipy.optimize import minimize

# ---- 시각화 설정 (환경 호환성) -------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')  # IDE/OS 환경에서 안정적인 백엔드
import matplotlib.pyplot as plt

# ---- Qiskit / Qiskit Nature ---------------------------------------------------
from qiskit.quantum_info import Statevector, SparsePauliOp  # SparsePauliOp: I-shift, .paulis/.coeffs 사용
from qiskit.primitives import StatevectorEstimator          # 이상적 기대값(무잡음) 계산용
from qiskit_nature.second_q.drivers import PySCFDriver      # PySCF로 분자 적분자/정보 취득
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.units import DistanceUnit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes

# ---- 전역 상수/하이퍼파라미터 ---------------------------------------------------
H2_BOND_LENGHT = 0.7414    # H2 평형 결합길이(Å) 근방
H2_BASIS = 'sto3g'
GLOBAL_SEED = 42

# VQD 패널티 β — 논문 수치(β=3 Ha)를 채택 (§5에서 H2 전 구간에 대해 사용)
PENALTY_BETA = 3.0

# 고전 최적화 설정 (논문 부록에서 다중 시도 권장 – 여기선 2회)
OPTIMIZATION_METHOD = 'Nelder-Mead'
ATTEMPT_COUNT = 2
CONVERGENCE_TOLERANCE = 1e-6
MAXIMUM_ITERATION = 10000

# RealAmplitudes(entanglement)
RA_ENTANGLEMENT = 'linear'

# 회로 도식 출력용 (선택적)
BLACKBOX_FIGSIZE = (25, 8)
BLACKBOX_FONTSIZE = 10
BLACKBOX_FOLD = -1
DECOMPOSED_FIGSIZE = (25, 8)
DECOMPOSED_FONTSIZE = 6
DECOMPOSED_FOLD = 30


# ------------------------- 유틸: 해밀토니안 사람-친화 출력 -------------------------
def print_hamiltonian(hamiltonian: SparsePauliOp) -> None:
    """
    SparsePauliOp의 각 Pauli 항과 계수를 사람이 읽기 쉽게 출력.
    - Qiskit의 SparsePauliOp는 .paulis (PauliList), .coeffs (복소 계수) 속성을 가짐.
      (IBM 문서: qiskit.quantum_info.SparsePauliOp API)  # 참고 근거
    """
    print(f"\n{'='*50}")
    print(f"H₂ 분자 해밀토니안 ({H2_BASIS} 기저)")
    print('='*50)

    coeffs = hamiltonian.coeffs
    paulis = hamiltonian.paulis

    print(f"총 Pauli 항 개수: {len(coeffs)}개")
    print(f"큐비트 개수: {hamiltonian.num_qubits}개")
    print("\n해밀토니안 식:")
    print("-" * 50)

    for coeff, pauli in zip(coeffs, paulis):
        c = float(coeff.real) if abs(coeff.imag) < 1e-12 else coeff  # 수치 안정적 실수 표현
        s = str(pauli)
        sign = "+" if (isinstance(c, complex) and c.real >= 0) or (not isinstance(c, complex) and c >= 0) else "-"
        mag = abs(c.real) if not isinstance(c, complex) else abs(c)  # 복소면 크기
        if sign == "+":
            print(f"+ {mag:.16f} * {s}")
        else:
            print(f"- {mag:.16f} * {s}")

    print("-" * 50)
    print(f"총 {len(coeffs)}개 항으로 구성된 해밀토니안")
    print('='*50)


# ------------------------- 1. H2 문제 생성 (PySCF → Nature) ----------------------
def create_h2_problem(bond_length: float = H2_BOND_LENGHT):
    """
    PySCFDriver로 H2 분자의 전자구조 문제(ElectronicStructureProblem)를 생성.
    - Nature는 전자 해밀토니안(전자항)만 연산자로 만들며, 핵-반발 E_nuc은
      “총 에너지” 계산에 상수로 더하는 것이 표준 흐름.
    """
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis=H2_BASIS,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()

    print(f"H2 분자 (Bond: {bond_length} Å)")
    print(f"전자 수: {problem.num_particles}")
    print(f"공간 오비탈: {problem.num_spatial_orbitals}")
    print(f"스핀 오비탈: {problem.num_spin_orbitals}")
    return problem


# ------------------------- 2. Ansatz: HF + RealAmplitudes ------------------------
class HFRealAmplitudesAnsatz:
    """
    Hartree–Fock 초기상태 + RealAmplitudes 파라메트릭 회로.

    · HF 초기상태: 점유 비트스트링 |1100> (예시) 등을 빠르게 준비하는 전처리 회로.
      (Qiskit Nature의 HartreeFock 회로)
    · RealAmplitudes: shallow/표준적 회로로, VQD의 목적함수 구조 검증에 충분.
      (논문은 UCCGSD 사용; 여기서는 편의상 대체. 알고리즘 수학은 동일.)

    반환 회로는 “분해된” 버전을 기본으로 사용:
      - blackbox_circuit: 컴포넌트 모듈 합성 상태
      - decomposed_circuit: 기본 게이트로 분해(시각화 시 가독성)
    """

    def __init__(self, num_spatial_orbitals, num_particles, mapper):
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.mapper = mapper
        self.num_parameters = None
        self.blackbox_circuit = None
        self.decomposed_circuit = None
        self.parameters = None

    def create_circuit(self, num_qubits: int) -> QuantumCircuit:
        # (a) HF 초기상태 |ψ_HF> 준비 회로
        hf_state = HartreeFock(
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_particles=self.num_particles,
            qubit_mapper=self.mapper
        )

        # (b) RealAmplitudes(파라메트릭) — reps=1, 선형 얽힘
        real_amplitudes = RealAmplitudes(
            num_qubits=num_qubits,
            reps=1,
            entanglement=RA_ENTANGLEMENT
        )

        # (c) 파라미터 백터 θ 생성 및 바인딩
        self.num_parameters = real_amplitudes.num_parameters
        self.parameters = ParameterVector('θ', self.num_parameters)
        real_amplitudes_with_params = real_amplitudes.assign_parameters(self.parameters)

        # (d) 블랙박스 회로(분해 없음)
        self.blackbox_circuit = QuantumCircuit(num_qubits)
        self.blackbox_circuit.compose(hf_state, inplace=True)
        self.blackbox_circuit.barrier()
        self.blackbox_circuit.compose(real_amplitudes_with_params, inplace=True)
        print(f"Blackbox circuit gates: {len(self.blackbox_circuit.data)}")

        # (e) 분해 회로(RealAmplitudes만 .decompose())
        self.decomposed_circuit = QuantumCircuit(num_qubits)
        self.decomposed_circuit.compose(hf_state, inplace=True)
        self.decomposed_circuit.barrier()
        self.decomposed_circuit.compose(real_amplitudes_with_params.decompose(), inplace=True)
        print(f"Decomposed circuit gates: {len(self.decomposed_circuit.data)}")

        print(f"파라미터 개수: {self.num_parameters}")
        print(f"파라미터 벡터: {self.parameters}")

        return self.decomposed_circuit


# ------------------------- 3. Variational Quantum Deflation ----------------------
class VariationalQuantumDeflation:
    """
    VQD 클래스 — 논문 §2–§4의 로직을 그대로 구현 (Ansatz만 RealAmplitudes 대체).

    목적함수:
      (식 (1))  E(λ)   = <ψ(λ)|H|ψ(λ)>
      (식 (2))  F(λ_k) = E(λ_k) + Σ_{i<k} β_i |<ψ(λ_k)|ψ(λ_i)>|^2

    해석적 관점(식 (4)):
      H_k = H + Σ_{i<k} β_i |i><i|   — k번째 상태는 H_k의 GS가 되도록 유도

    오버랩(§4):
      |<ψ(λ_i)|ψ(λ_k)>|^2 = |<0| R(λ_i)^† R(λ_k) |0>|^2
      · 회로: R(λ_k) → (R(λ_i))^† 를 붙여 |0…0> 확률을 측정(본 코드는 Statevector로 정확 계산)

    샘플링/깊이:
      · 논문은 기존 VQE와 동일 큐빗 수, 최대 약 2배 깊이로 구현 가능함을 강조(§4).
      · 본 코드는 이상적 기대값(무잡음) 모드(StatevectorEstimator)로 구조 검증에 집중.
    """

    def __init__(self, hamiltonian: SparsePauliOp, problem, beta=PENALTY_BETA, seed=GLOBAL_SEED, exact_energies=None):
        self.H = hamiltonian
        self.problem = problem
        self.beta = beta
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.exact_energies = exact_energies

        # Ansatz 준비 (HF + RealAmplitudes)
        mapper = JordanWignerMapper()
        self.ansatz = HFRealAmplitudesAnsatz(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper
        )

        self.circuit = self.ansatz.create_circuit(self.H.num_qubits)

        # 이상적 기대값 추정기 — Pauli 관측자(SparsePauliOp)에 대해 기대값 제공
        # (IBM 문서: qiskit.primitives.StatevectorEstimator)
        self.estimator = StatevectorEstimator(seed=seed)

        # 누적 저장소
        self.energies = []
        self.params_list = []

        print("VQD 초기화 완료")

    # --------- (식 (1)) E(λ) = <ψ(λ)|H|ψ(λ)> ----------
    def _compute_energy(self, params: np.ndarray) -> float:
        circ = self.circuit.assign_parameters(params)
        result = self.estimator.run([(circ, [self.H])]).result()
        return float(result[0].data.evs[0])

    # --------- (§4) |<ψ_i|ψ_k>|^2 = |<0| R(λ_i)^† R(λ_k) |0>|^2 ----------
    def _compute_overlap(self, params_k: np.ndarray, params_i: np.ndarray) -> float:
        # R(λ_k)
        circuit_k = self.circuit.assign_parameters(params_k)
        # R(λ_i)^†
        circuit_i_inv = self.circuit.assign_parameters(params_i).inverse()

        # 오버랩 회로: R(λ_i)^† R(λ_k) |0>
        overlap_circuit = QuantumCircuit(self.H.num_qubits)
        overlap_circuit.compose(circuit_k, inplace=True)
        overlap_circuit.compose(circuit_i_inv, inplace=True)

        # 이상적 상태벡터로 |0…0> 진폭의 제곱을 정확 계산 (논문 §4의 샷 기반 방법과 수학적으로 등가)
        state = Statevector(overlap_circuit)
        zero_prob = abs(state[0]) ** 2
        return float(min(max(zero_prob, 0.0), 1.0))

    # --------- (식 (2)) Ground-state 단계: F = E ----------
    def _ground_state_objective(self, params: np.ndarray) -> float:
        return self._compute_energy(params)

    # --------- (식 (2)) k≥1 단계: F = E + Σ β |overlap|^2 ----------
    def _vqd_objective(self, params: np.ndarray) -> float:
        energy = self._compute_energy(params)
        penalty = 0.0
        for prev_params in self.params_list:
            overlap = self._compute_overlap(params, prev_params)
            penalty += self.beta * overlap
        return energy + penalty

    # --------- VQE로 GS(상태 0) ----------
    def find_ground_state(self) -> bool:
        print("Ground State (State 0) 계산 중...")
        best_energy = np.inf
        best_params = None

        for _ in range(ATTEMPT_COUNT):
            theta0 = self.rng.random(self.ansatz.num_parameters)
            result = minimize(
                self._ground_state_objective,
                theta0,
                method=OPTIMIZATION_METHOD,
                options={'xatol': CONVERGENCE_TOLERANCE, 'fatol': CONVERGENCE_TOLERANCE,
                         'maxiter': MAXIMUM_ITERATION, 'disp': True}
            )
            if result.success and result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()

        self.energies.append(best_energy)
        self.params_list.append(best_params)

        if self.exact_energies is not None:
            exact_energy = self.exact_energies[0]
            error = abs(best_energy - exact_energy)
            print(f"State 0 완료 - 에너지: {best_energy:.8f} Ha")
            print(f"State 0 오차: {error:.3e} Ha ({error*1e6:.3f} µHa)")
        else:
            print(f"State 0 완료 - 에너지: {best_energy:.8f} Ha")
        return True

    # --------- VQD로 들뜬 상태(상태 1,2,...) ----------
    def find_excited_state(self) -> bool:
        state_idx = len(self.energies)
        print(f"State {state_idx} 계산 중...")

        best_energy = np.inf
        best_params = None

        for _ in range(ATTEMPT_COUNT):
            theta0 = self.rng.random(self.ansatz.num_parameters)
            result = minimize(
                self._vqd_objective,
                theta0,
                method=OPTIMIZATION_METHOD,
                options={'xatol': CONVERGENCE_TOLERANCE, 'fatol': CONVERGENCE_TOLERANCE,
                         'maxiter': MAXIMUM_ITERATION, 'disp': True}
            )
            if result.success:
                # 주의: 목적함수에는 오버랩 패널티가 포함되어 있으므로,
                # 순수 에너지 평가는 따로 계산해 비교(논문 Figure 2와 동일 지표)
                final_energy = self._compute_energy(result.x)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_params = result.x.copy()

        self.energies.append(best_energy)
        self.params_list.append(best_params)

        if self.exact_energies is not None and state_idx < len(self.exact_energies):
            exact_energy = self.exact_energies[state_idx]
            error = abs(best_energy - exact_energy)
            print(f"State {state_idx} 완료 - 에너지: {best_energy:.8f} Ha")
            print(f"State {state_idx} 오차: {error:.3e} Ha ({error*1e6:.3f} µHa)")
        else:
            print(f"State {state_idx} 완료 - 에너지: {best_energy:.8f} Ha")
        return True

    # --------- 회로 도식 출력(옵션) ----------
    def draw_circuit_blackbox(self, state_idx=0, title=None):
        if state_idx >= len(self.params_list):
            print(f"State {state_idx} does not exist. Only {len(self.params_list)} states calculated.")
            return None

        params = self.params_list[state_idx]
        param_circuit = self.ansatz.blackbox_circuit.assign_parameters(params)

        figsize, fold, fontsize = BLACKBOX_FIGSIZE, BLACKBOX_FOLD, BLACKBOX_FONTSIZE
        if title is None:
            title = f"VQD State {state_idx} Circuit (Blackbox, E = {self.energies[state_idx]:.6f} Ha)"

        num_gates = len(param_circuit.data)
        print(f"State {state_idx} blackbox gates: {num_gates}, Figure size: {figsize}")

        circuit_fig = param_circuit.draw(
            output='mpl',
            fold=fold,
            style={'dpi': 600, 'fontsize': fontsize, 'subfontsize': fontsize - 2}
        )
        circuit_fig.set_size_inches(figsize[0], figsize[1])
        circuit_fig.suptitle(title, fontsize=fontsize + 4, fontweight='bold')

        filename = f"vqd_circuit_state_{state_idx}_blackbox.png"
        circuit_fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"VQD State {state_idx} blackbox circuit saved as '{filename}'")
        plt.show()
        return circuit_fig

    def draw_circuit_decomposed(self, state_idx=0, title=None):
        if state_idx >= len(self.params_list):
            print(f"State {state_idx} does not exist. Only {len(self.params_list)} states calculated.")
            return None

        params = self.params_list[state_idx]
        param_circuit = self.circuit.assign_parameters(params)

        figsize, fold, fontsize = DECOMPOSED_FIGSIZE, DECOMPOSED_FOLD, DECOMPOSED_FONTSIZE
        if title is None:
            title = f"VQD State {state_idx} Circuit (Decomposed, E = {self.energies[state_idx]:.6f} Ha)"

        num_gates = len(param_circuit.data)
        print(f"State {state_idx} decomposed gates: {num_gates}, Figure size: {figsize}, Fold: {fold}")

        circuit_fig = param_circuit.draw(
            output='mpl',
            fold=fold,
            style={'dpi': 600, 'fontsize': fontsize, 'subfontsize': fontsize - 1}
        )
        circuit_fig.set_size_inches(figsize[0], figsize[1])
        circuit_fig.suptitle(title, fontsize=fontsize + 4, fontweight='bold')

        filename = f"vqd_circuit_state_{state_idx}_decomposed.png"
        circuit_fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"VQD State {state_idx} decomposed circuit saved as '{filename}'")
        plt.show()
        return circuit_fig

    # --------- VQD 전체 실행 ----------
    def run(self, num_states=6):
        print(f"VQD 실행 - {num_states}개 상태 계산")
        self.find_ground_state()
        for _ in range(1, num_states):
            self.find_excited_state()
        return self.energies


# ------------------------- 4. 결과 분석/시각화 ------------------------------------
def analyze_results(vqd_energies, exact_energies):
    """
    VQD 결과 vs. 정확치 비교.
    - 논문 Figure 2처럼 각 상태별 에너지를 비교하고, µHa 척도 오차를 표시.
    - 논문 성능지표: median error < 4 µHa (H2 전 구간, UCCGSD 사용) — 여기선 동일 임계로 출력만 확인.
    """
    vqd_energies = np.array(vqd_energies)
    exact_energies = np.array(exact_energies[:len(vqd_energies)])
    errors = vqd_energies - exact_energies

    print("\n=== 결과 분석 ===")
    print(" State |   VQD (Ha)   |  Exact (Ha)  |  Error (Ha)  | Error (µHa)")
    print("-------|--------------|--------------|--------------|------------")

    for i, (vqd, exact, error) in enumerate(zip(vqd_energies, exact_energies, errors)):
        error_muha = abs(error) * 1e6
        print(f" {i:>4d} | {vqd:12.8f} | {exact:12.8f} | {error:+12.8f} | {error_muha:10.3f}")

    median_error = np.median(np.abs(errors))
    max_error = np.max(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    print("-------|--------------|--------------|--------------|------------")
    print(f"중간 오차: {median_error:.3e} Ha ({median_error*1e6:.3f} µHa)")
    print(f"최대 오차: {max_error:.3e} Ha ({max_error*1e6:.3f} µHa)")
    print(f"RMSE: {rmse:.3e} Ha ({rmse*1e6:.3f} µHa)")

    chemical_accuracy = 1.6e-3
    paper_target = 4e-6
    print(f"\n화학적 정확도 (<1.6e-3 Ha): {'✅' if median_error < chemical_accuracy else '❌'}")
    print(f"논문 목표 (<4e-6 Ha): {'✅' if median_error < paper_target else '❌'}")

    return {
        'energies': vqd_energies,
        'exact': exact_energies,
        'errors': errors,
        'median_error': median_error,
        'max_error': max_error,
        'rmse': rmse
    }


def create_visualization(results):
    """
    두 패널 그림:
      (좌) 에너지 준위 비교  (우) µHa 오차 및 4 µHa 목표 밴드
    """
    if results is None:
        print("결과가 없어 시각화를 건너뜁니다.")
        return

    energies = results['energies']
    exact = results['exact']
    errors = results['errors']
    states = np.arange(len(energies))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(states, energies, 'ro-', markersize=10, linewidth=2.5,
             markeredgewidth=1.5, markeredgecolor='darkred', label='VQD')
    ax1.plot(states, exact, 'b--', linewidth=2.5, alpha=0.8, label='Exact')
    ax1.set_xlabel('State Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (Ha)', fontsize=12, fontweight='bold')
    ax1.set_title('H₂ Energy Level Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(states, errors * 1e6, 'ro-', markersize=10, linewidth=2.5,
             markeredgewidth=1.5, markeredgecolor='darkred', label='Error')
    ax2.axhline(4, color='red', linestyle=':', linewidth=2, label='Paper Target (4 µHa)')
    ax2.axhline(-4, color='red', linestyle=':', linewidth=2)
    ax2.fill_between(states, -4, 4, alpha=0.2, color='green', label='Target Achievement')
    ax2.set_xlabel('State Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (µHa)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Error Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = 'vqd_results_micro.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"시각화 결과가 '{output_filename}'로 저장되었습니다.")
    plt.show()
    plt.close()


def draw_quantum_circuits(vqd: VariationalQuantumDeflation):
    """
    (옵션) 상태 0 회로만 블랙박스/분해본으로 그림 저장
    """
    print(f"\n{'='*30}")
    print("양자 회로 출력 (State 0만)")
    print('='*30)

    print(f"State 0 회로 출력:")
    print(f"     - 블랙박스 회로...")
    vqd.draw_circuit_blackbox(0)
    print(f"     - 분해된 회로...")
    vqd.draw_circuit_decomposed(0)

    print(f"\n양자 회로 출력 완료!")
    print(f"- 블랙박스 회로: State 0")
    print(f"- 분해된 회로: State 0")
    print(f"- 총 2개 회로 파일이 생성되었습니다.")


# ------------------------- 5. 메인 루틴 ------------------------------------------
def main():
    print("="*60)
    print("H₂ 분자 VQD 계산 (HF + RealAmplitudes)")
    print("="*60)

    # (1) 분자 문제 생성 (전자 해밀토니안 + 핵-반발 상수 정보 보유)
    problem = create_h2_problem(bond_length=H2_BOND_LENGHT)

    # (2) 해밀토니안 매핑 — 전자 해밀토니안만 연산자로 생성
    mapper = JordanWignerMapper()
    H_elec: SparsePauliOp = mapper.map(problem.hamiltonian.second_q_op())

    # (★수정 1) 핵-반발 에너지 E_nuc를 정체성 항으로 shift하여 총 해밀토니안 구성
    #  · H = H_elec + E_nuc * I
    #  · SparsePauliOp에서 I 항을 명시적으로 만들어 더한다.
    e_nuc = problem.hamiltonian.nuclear_repulsion_energy
    I = SparsePauliOp.from_list([("I" * H_elec.num_qubits, 1.0)])
    H: SparsePauliOp = H_elec + e_nuc * I

    print(f"\n해밀토니안 정보:")
    print(f"Pauli 항: {len(H.paulis)}개")
    print(f"큐비트: {H.num_qubits}개")

    # (3) 해밀토니안 상세 출력 (사람-친화)
    print_hamiltonian(H)

    # (4) 정확 고유값 (정확 대각화) — 논문 Figure 2의 'Exact'에 해당
    exact_energies = np.sort(np.linalg.eigvalsh(H.to_matrix()))
    print(f"\n정확한 고유값 (6개):")
    for i, energy in enumerate(exact_energies[:6]):
        print(f"  State {i}: {energy:.8f} Ha")

    # (5) VQD 실행 (상태 0 → k 순차)
    print(f"\n{'='*30}")
    print("VQD 단계 (Ground state부터)")
    print('='*30)

    vqd = VariationalQuantumDeflation(H, problem, beta=PENALTY_BETA, seed=GLOBAL_SEED, exact_energies=exact_energies)
    all_energies = vqd.run(num_states=6)

    # (6) 결과 분석 (µHa 지표 포함)
    print(f"\n{'='*30}")
    print("결과 분석")
    print('='*30)
    results = analyze_results(all_energies, exact_energies)

    # (7) 시각화
    print(f"\n{'='*30}")
    print("결과 시각화")
    print('='*30)
    create_visualization(results)

    # (8) 회로 도식 (State 0)
    draw_quantum_circuits(vqd)

    print(f"\n{'='*60}")
    print("H₂ 분자 VQD 계산 완료 (HF + RealAmplitudes)")
    print("- 블랙박스 회로: 컴포넌트들로 구성")
    print("- 분해된 회로: 기본 게이트들로 분해")
    print("="*60)


if __name__ == "__main__":
    main()
