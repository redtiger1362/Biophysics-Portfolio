import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 시스템 파라미터 및 평형점 해석적 계산
alpha, beta = 0.5, 0.02
delta, gamma = 0.01, 0.3
K = 100.0  # 환경 수용력
# day 4-2와 비교할때 토끼 번식률 5배 증가, 여우 자연 폐사율 3배 증가

# Non-trivial Fixed point 계산
# dxdt = alpha * x * (1 - x / K) - beta * x * y
# dydt = delta * x * y - gamma * y
# 평형점은 dxdt = 0, dydt = 0을 만족하는 점
# dydt = 0 => delta * x * y - gamma * y = 0 => y*(delta*x - gamma) = 0
# y=0 또는 x = gamma/delta, 여기서 y=0이라는 여우가 모두 멸종한 trivial solution은 제외
# 따라서 x* = gamma/delta
# dxdt = 0 => alpha * x * (1 - x / K) - beta * x * y = 0
# x*(1 - x*/K) = (beta/alpha) * x* *y* => 1 - x*/K = (beta/alpha) * y* 
# => y* = (alpha/beta) * (1 - x*/K)

x_star = gamma / delta
y_star = alpha / beta * (1 - x_star / K)
print(f"Theoretical Fixed Point: (x*, y*) = ({x_star:.2f}, {y_star:.2f})")

# 2. [이론 1] 동역학계 Jacobian Matrix 및 고유값 분석
# 평형점 근처(아주 약간 벗어났을때)에서 시스템이 어떻게 반응하는지 확인하기 위해 Jacobian Matrix 계산
# dxdt를 f, dydt를 g라고 하면 J = [[df/dx, df/dy], [dg/dx, dg/dy]]
J11 = alpha * (1 - 2 * x_star / K) - beta * y_star
J12 = -beta * x_star
J21 = delta * y_star
J22 = delta * x_star - gamma

Jacobian = np.array([[J11, J12], [J21, J22]])
eigenvalues = np.linalg.eigvals(Jacobian)

print(f"Eigenvalues of the Jacobian at the fixed point: {eigenvalues}")
if all(np.real(eigenvalues) < 0):
    print("Because all eigenvalues have negative real parts, the fixed point is stable.")
    # stable spiral sink

# 3. [이론2] '확률론적' 랑제뱅 역학 (Euler-Maruyama method)
t_end = 200
dt = 0.05
steps = int(t_end / dt)
t = np.linspace(0, t_end, steps)

# 앙상블 배열 초기화
x_stock = np.zeros(steps)
y_stock = np.zeros(steps)
x_stock[0], y_stock[0] = 40.0, 10.0  # 초기 조건

# 내재적 요동 (demographic Noise) 강도
sigma_noise = 0.15

for i in range(1, steps):
    dx_det = (alpha * x_stock[i-1] * (1 - x_stock[i-1] / K) - beta * x_stock[i-1] * y_stock[i-1]) * dt
    dy_det = (delta * x_stock[i-1] * y_stock[i-1] - gamma * y_stock[i-1]) * dt
    # 기존의 dxdt,dydt와 달리 확률론적 요동을 추가하여 Euler-Maruyama 방법으로 업데이트함

    # 확률론적 요동(Stochastic diffusion) - 개체수의 제곱근에 비례하는 통계역학적 잡음
    noise_x = sigma_noise * np.sqrt(max(x_stock[i-1], 0)) * np.random.normal(0, np.sqrt(dt))
    noise_y = sigma_noise * np.sqrt(max(y_stock[i-1], 0)) * np.random.normal(0, np.sqrt(dt))
    # max(x_stock[i-1], 0)사용하여 음수 개체수 방지, 루트 내부에서 음수가 되면 계산불가하므로 0으로 변환
    # np.random.normal(0, np.sqrt(dt))는 평균 0, 분산 dt인 정규분포 난수 생성
    # 브라운 운동에서는 시간 간격의 제곱근에 해당하는 변화가 발생

    x_stock[i] = x_stock[i-1] + dx_det + noise_x  # 첫째 항은 평균적인 변화, 둘째 항은 무작위 변화
    y_stock[i] = y_stock[i-1] + dy_det + noise_y

# '결정론적' 비교용 기준선, 기존은 Euler-Maruyama로 계산했고, 이제는 Scipy의 solve_ivp를 사용하여 결정론적 ODE를 풀어 비교
def deterministic_ode(t, z):  # 잡음이 완전이 제거된 미분방정식
    x, y  = z
    return [alpha * x * (1 - x / K) - beta * x * y, delta * x * y - gamma * y]
    # Scipy solve_ivp는 함수의 반환값이 리스트나 배열이어야 하며 현재의 위치가 아닌 위치에서의 변화율을 필요로 함

sol_det = solve_ivp(deterministic_ode, [0, t_end], [40.0, 10.0], t_eval=t, method='RK45')

# 4. phase space 시각화
fig, ax  = plt.subplots(figsize=(10,8))  # figure는 전체 캔버스, axes는 실제 그래프를 그리는 공간
fig.suptitle(' Phase space: Deterministic Attractior vs Stochastic Quasi-cycles', fontsize=16, fontweight='bold')

# 결정론적 궤적
ax.plot(sol_det.y[0], sol_det.y[1], color='black', lw=2, label='Deterministic ODE')
# 확률론적 궤적
ax.plot(x_stock, y_stock, color='teal', lw=1, alpha=0.7, label='Stochastic SDE (Demographic Noise)')

# 이론적 평형점 표시
ax.scatter(x_star, y_star, color='red', s=100, label=f'Theoretical Fixed Point\nEigenvals: {eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}', zorder=5)

ax.set_title('Linear Stability and Fluctuation in Population Dynamics')
ax.set_xlabel('Prey Population (x)', fontsize=12)
ax.set_ylabel('Predator Population (y)', fontsize=12)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()  # 자동 레이아웃 조정
plt.show()