import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp # 연구실 표준 적분기

# 물리 상수
m, k, gamma = 1.0, 10.0, 0.5

#미분방정식을 함수 형태로 정의 (연구실 표준)
def damped_oscillator(t, state):
    x, v = state
    dxdt = v
    dvdt = (-k * x - gamma * v) / m
    return [dxdt, dvdt]

# 초기 조건 및 시간 구간
state0 = [1.0, 0.0]  # 초기 위치와 속도
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)  # 적분을 위한 시간 점들

# [핵심] Scipy의 solve_ivp를 사용하여 미분방정식 수치해석 수행
sol = solve_ivp(damped_oscillator, t_span, state0, t_eval=t_eval, method='RK45')  # RK45는 Runge-Kutta 방법 중 하나

# 비교용 이론값
exact_sol = 1.0 * np.exp(-gamma*t_eval/(2*m)) * np.cos(np.sqrt(k/m)*t_eval)

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Numerical Simulation (Scipy solve_ivp)', color='red', linestyle='--')
plt.plot(t_eval, exact_sol, label='Analytical Solution (Exact Formula)', color='blue', alpha=0.5)
plt.title('Validation: Numerical vs Analytical Solution of Damped Harmonic Oscillation (Scipy)')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (x)')
plt.axhline(0, color='black', lw=1)
plt.grid(True)
plt.legend()
plt.show()
