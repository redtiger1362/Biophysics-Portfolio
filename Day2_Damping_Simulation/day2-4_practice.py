import numpy as np
import matplotlib.pyplot as plt

# 1. 물리 상수 및 시뮬레이션 파라미터 설정
m = 1.0 
k = 10.0
gamma = 0.5

# 수치 해석 핵심: 시간 간격(dt)설정
t_start, t_end, dt = 0, 10, 0.01
t = np.arange(t_start, t_end, dt)
n_steps = len(t)

# 2. 상태 변수(x, v) 초기화
x_num = np.zeros(n_steps)  # 위치 배열 초기화
v_num = np.zeros(n_steps)  # 속도 배열 초기화

# 초기 조건 설정(t=0에서 초기 위치와 속도)
x_num[0] = 1.0  # 초기 위치
v_num[0] = 0.0  # 초기 속도

# 3. [핵심] Euler 방법을 사용하여 감쇠진동 시스템의 수치 해석 수행
for n in range(n_steps - 1):
    # 가속도 계산: a = (-k*x - gamma*v) / m
    a = (-k * x_num[n] - gamma * v_num[n]) / m
    # Euler 방법으로 다음 위치와 속도 계산
    v_num[n + 1] = v_num[n] + a * dt # 속도를 먼저 업데이트
    x_num[n + 1] = x_num[n] + v_num[n] * dt # 방금 구한 새 속도로 위치 업데이트

# 4. 검증을 위한 해석적 해 계산
omega = np.sqrt(k/m)
x_exact = 1.0 * np.exp(-gamma*t/(2*m)) * np.cos(omega*t)

# 5. 수치해석 결과와 해석적 해 비교 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(t, x_num, label='Numerical Simulation (Euler Method)', color='red', linestyle='--')
plt.plot(t, x_exact, label='Analytical Solution (Exact Formula)', color='blue', alpha=0.5)

plt.title('Validation: Numerical vs Analytical Solution of Damped Harmonic Oscillation')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (x)')
plt.axhline(0, color='black', lw=1)
plt.grid(True)
plt.legend()
plt.show()


#Euler-Cromer 방법을 사용하여 과거의 속도를 쓰지 않고 현재의 속도를 사용하여 오차를 줄임
#그러나 Euler-Cromer 방법은 에너지 보존이 잘 되지 않음
# 1차 근사법이므로 오차가 누적될 수 있음
# 따라서 더 정확한 수치해석을 위해서는 Scipy 같은 라이브러리를 사용하는 것이 좋음(day2-5_practice.py에서 다룸)
