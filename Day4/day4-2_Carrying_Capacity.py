import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 생태계 파라미터 (Rates)
alpha = 0.1   # 토끼의 번식률
beta = 0.02   # 여우가 토끼를 잡아먹는 비율
delta = 0.01  # 여우가 토끼를 먹고 번식하는 효율
gamma = 0.1   # 여우의 자연 폐사율
K = 250.0     # [추가됨] 토끼의 환경 수용력 (풀의 양 한계)

# 2. 로지스틱 성장이 추가된 미분방정식 정의
def lotka_volterra(t, state):
    x, y = state  # x: 토끼(Prey), y: 여우(Predator)
    
    # 토끼의 변화율 = (수용력 K를 반영한 로지스틱 번식) - (여우에게 잡아먹힘)
    dxdt = alpha * x * (1 - x / K) - beta * x * y
    # 여우의 변화율 = (토끼를 먹고 번식) - (자연 폐사)
    dydt = delta * x * y - gamma * y
    
    return [dxdt, dydt]

# 3. 초기 조건 및 시간 설정
t_span = (0, 500)               # [수정] 수렴하는 모습을 보기 위해 500일로 연장
t_eval = np.linspace(0, 500, 2000)
initial_state = [40.0, 9.0]     # 초기 토끼 40마리, 여우 9마리

# 4. 수치 해석
solution = solve_ivp(lotka_volterra, t_span, initial_state, t_eval=t_eval)

# 데이터 추출
t = solution.t
rabbits = solution.y[0]
foxes = solution.y[1]

# 5. 시계열(Time-series) 시각화
plt.figure(figsize=(10, 5))
plt.plot(t, rabbits, label='Rabbits (Prey)', color='blue', linewidth=2)
plt.plot(t, foxes, label='Foxes (Predator)', color='red', linewidth=2)

plt.title('Lotka-Volterra Predator-Prey Dynamics', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# --- [추가 코드] ---
# 6. 위상 공간(Phase Space) 시각화
plt.figure(figsize=(6, 6))
plt.plot(rabbits, foxes, color='purple', linewidth=1.5)

# 출발점 표시
plt.scatter(rabbits[0], foxes[0], color='black', s=50, label='Start Point', zorder=5)

plt.title('Phase Space Portrait (Rabbit vs Fox)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Rabbits')
plt.ylabel('Number of Foxes')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
