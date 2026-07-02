import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 생태계 파라미터
alpha = 0.1  # 토끼 번식률
beta = 0.02  # 여우가 토끼를 잡아먹는 비율
delta = 0.01 # 여우가 토끼를 먹고 번식하는 효율
gamma = 0.1  # 여우 자연사율

# 2. 로트카-볼테라 연립 미분방정식
def lotka_volterra(t, state):
    x, y = state #x: 토끼 개체수, y: 여우 개체수
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# 3. 초기 조건 및 시간 설정
t_span = (0, 200) # 시뮬레이션 시간 범위(0~200)
t_eval = np.linspace(0,200, 1000)
initial_state = [40.0, 9.0] # 초기 토끼 개체수, 초기 여우 개체수

# 4. 수치 해석 (Scipy solve_ivp사용)
solution = solve_ivp(lotka_volterra, t_span, initial_state, t_eval=t_eval)

# 데이터 추출
t = solution.t
rabbits = solution.y[0]
foxes = solution.y[1]

# 5. Time series 시각화
plt.figure(figsize=(10,5))
plt.plot(t, rabbits, label='Rabbits', color='blue')
plt.plot(t, foxes, label='Foxes', color='red')

plt.title('Lotka-Volterra Model: Predator-Prey Dynamics',fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 6. Phase space 시각화
plt.figure(figsize=(6,6))
plt.plot(rabbits, foxes, color='purple', lw=1.5)

# 출발점 표시
plt.scatter(rabbits[0], foxes[0], color='black', s=100, label='Start', zorder=5)
plt.title('Phase Space: Predator-Prey Dynamics',fontsize=14, fontweight='bold')
plt.xlabel('Rabbits Population', fontsize=12)
plt.ylabel('Foxes Population', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# Figure1에서는 파란색 선(토끼)가 증가한 후 붉은색 선(여우)가 뒤따라 증가
# Figure2에서는 그래프가 원을 그리며 제자리로 돌아와는 모습을 확인할 수 있으며 
# 이는 포식자 피식자 관계가 외부 개입없이 동일한 주기로 반복됨