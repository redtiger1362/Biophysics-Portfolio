import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# 1. 연구 환경 및 초기 조건 세팅
m = 1.0 # 특정 단백질 복합체 질량
k = 10.0 # 세포 내 탄성 상수

#비교할 생물학적 유체 환경
environments = {
    'Water (In vitro)': 0.1,
    'Cytoplasm': 0.5, 
    'Protein Condensate (Dense)': 2.0 
}

t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 1500)
initial_state = [1.0, 0.0]  # 초기 위치와 속도

#미분 방정식 (상태공간 모델)
def biophysics_model(t, state, gamma):
    x, v = state
    dxdt = v
    dvdt = (-k * x - gamma * v) / m
    return [dxdt, dvdt]

# 2. 다중 패널 대시보드 도화지 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Biophysical Dynamics of Damped Harmonic oscillator across Media', fontsize=16, fontweight='bold')

summary_results = []

print("="*60)
print("Biophysical Dynamics Simulation Results")
print("="*60)

# 3. 시뮬레이션 루프 및 물리량 계산
for env_name, gamma in environments.items():
    # Scipy를 이용한 수치해석
    sol = solve_ivp(biophysics_model, t_span, initial_state, t_eval=t_eval, args=(gamma,), method='RK45')
    t= sol.t
    x = sol.y[0]
    v = sol.y[1]

    # 물리량 계산 (운동에너지 + 위치에너지 = 총 에너지)
    E_kinetic = 0.5 * m * v**2
    E_potential = 0.5 * k * x**2
    E_total = E_kinetic + E_potential

    # 무차원 감쇠비(Damping Ratio, zeta) 계산
    zeta = gamma / (2 * np.sqrt(k * m))

    # 데이터 기록
    summary_results.append({  
        'Environment': env_name,
        'Gamma (Damping)': gamma,
        'Damping Ratio (ζ)': round(zeta, 3),
        'Final Energy Residual': round(E_total[-1], 5),
        'Max Velocity': round(np.max(np.abs(v)), 3)
    })

    # [패널 1] 시간에 따른 위치 변화 (Time Domain)
    axes[0].plot(t, x, label=f'{env_name}', linewidth=1.5)

    # [패널 2] 위상 공간 궤적 (Phase Space: x vs v)
    axes[1].plot(x, v, label=f'{env_name}', linewidth=1.5)

    # [패널 3] 역학적 에너지 소산(Energy Dissapation)
    axes[2].plot(t, E_total, label=f'{env_name}', linewidth=1.5)

# 4. 그래프 꾸미기
axes[0].set_title('Displacement over Time')
axes[0].set_xlabel('Time (t)')
axes[0].set_ylabel('Displacement (x)')
axes[0].axhline(0, color='black', lw=0.8, linestyle='--')

axes[1].set_title('Phase Space Trajectory(x vs v)')
axes[1].set_xlabel('Displacement (x)')
axes[1].set_ylabel('Velocity (v)')
axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
axes[1].axvline(0, color='black', lw=0.8, linestyle='--')

axes[2].set_title('Total Mechanical Energy Decay over Time')
axes[2].set_xlabel('Time (t)')
axes[2].set_ylabel('Total Energy (E)')
axes[2].axhline(0, color='black', lw=0.8, linestyle='--')

# 범례 및 레이아웃 조정
for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 5. 종합 리포트 생성 및 CSV 추출
df_report = pd.DataFrame(summary_results)
csv_filename = 'Comprehensive_Report.csv'
df_report.to_csv(csv_filename, index=False)

print("-" * 60)
print(f"Comprehensive report saved to {csv_filename}")
print("-" * 60)
print(df_report.to_string(index=False))
print("="*60)

plt.show()