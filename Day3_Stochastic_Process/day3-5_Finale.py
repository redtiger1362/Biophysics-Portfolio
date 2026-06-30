import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 융합 시스템 파라미터 세팅
n_steps = 15000
dt = 0.02
t = np.arange(n_steps)* dt

m= 1.0
k= 3.0
gamma = 1.0 # 점성 감쇠 계수
sigma = 0.5 # 배경 열 잡음(브라운운동)
F_run = 4.0

# 피드백 파라미터
p_base = 0.02 # 텀블링 확률(안정 상태에서 2%)
beta = 0.15 # 위치에너지에 따른 텀블링 피드백 민감도

# 상태 저장 배열
x,y = np.zeros(n_steps), np.zeros(n_steps)
vx,vy = np.zeros(n_steps), np.zeros(n_steps)
theta = np.zeros(n_steps)
p_tumble_history = np.zeros(n_steps)
Ep_history = np.zeros(n_steps)

theta[0] = np.random.uniform(0,2*np.pi)

# 2. 피드백이 결합된 능동-수동 확률미분방정식 해석
for i in range(1,n_steps):
    # 현재 입자의 위치에너지 계산
    Ep= 0.5 * k * (x[i-1]**2 + y[i-1]**2)
    Ep_history[i] = Ep

    #위치 에너지 비례 텀블링 확률 증가(최대 1.0 제한)
    current_p_tumble = np.clip(p_base + beta * Ep, 0.0, 1.0)
    p_tumble_history[i] = current_p_tumble

    # 결정된 실시간 확률을 바탕으로 Tumble 여부 결정
    if np.random.rand() < current_p_tumble:
        theta[i] = np.random.uniform(0,2*np.pi)
    else:
        theta[i] = theta[i-1]
    
    # 힘 계산
    Fx = -k * x[i-1] - gamma * vx[i-1] + F_run * np.cos(theta[i])
    Fy = -k * y[i-1] - gamma * vy[i-1] + F_run * np.sin(theta[i])

    # 열 잡음 생성
    noise_x = np.random.normal(0, np.sqrt(dt))
    noise_y = np.random.normal(0, np.sqrt(dt))

    # 가속도 계산 및 상태 업데이트(Euler_Maruyama)
    vx[i] = vx[i-1] + (Fx/m) * dt + (sigma / m) * noise_x
    vy[i] = vy[i-1] + (Fy/m) * dt + (sigma / m) * noise_y

    x[i] = x[i-1] + vx[i] * dt
    y[i] = y[i-1] + vy[i] * dt

# 3. 피드백 상호작용 시각화
fig = plt.figure(figsize=(16,10))
fig.suptitle('Coupled Dynamics: Mechanosensitive Tumbling in a Harmonic Potential', 
             fontsize=18, fontweight='bold')

# [패널1] 텀블링 확률 매핑 2D궤적
ax1 = plt.subplot(2,2,1)
#궤적을 텀블링 확률에 따라 색상 매핑
scatter = ax1.scatter(x,y, c= p_tumble_history, cmap='jet', s=5, alpha=0.6)
ax1.set_title('Spatial Trajectory (Color: Tumble Probability)', fontsize=12)
ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Position (y)')
ax1.axhline(0, color='black', lw=0.5, ls='--')
ax1.axvline(0, color='black', lw=0.5, ls='--')
ax1.axis('equal')
plt.colorbar(scatter, ax= ax1, label='Instantaneous Tumble Probability')

# [패널2] 위상 공간의 창발적 거동
ax2= plt.subplot(2,2,2)
#기존 day2-6의 나선형 소용돌이가 능동력과 피드백에 의해 파괴 및 재구성
ax2.plot(x,vx,color='gray', alpha=0.3, lw=0.5)
scatter2 = ax2.scatter(x,vx,c=p_tumble_history, cmap='plasma', s=2, alpha=0.8)
ax2.set_title('Phase Portrait (x vs vx) colored by Tumble Prob.', fontsize=12)
ax2.set_xlabel('Position (x)')
ax2.set_ylabel('Velocity (vx)')
ax2.grid(True, ls=':', alpha=0.5)

# [패널 3] 피드백 상관관계 (Correlation Plot) 직접 증명
ax3 = plt.subplot(2, 2, 3)
ax3.scatter(Ep_history, p_tumble_history, color='teal', alpha=0.1, s=10)
# 이론적 피드백 설계선 표시
e_vals = np.linspace(0, max(Ep_history), 100)
p_vals = np.clip(p_base + beta * e_vals, 0, 1)
ax3.plot(e_vals, p_vals, color='red', linestyle='--', linewidth=2, label=r'Design: $p = p_{base} + \beta E_p$')
ax3.set_title('Feedback Validation: Potential Energy vs Tumble Probability', fontsize=12)
ax3.set_xlabel('Potential Energy ($E_p$)')
ax3.set_ylabel('Tumble Probability ($p_{tumble}$)')
ax3.legend()
ax3.grid(True, ls=':', alpha=0.5)

# [패널 4] 시계열 피드백 작용 (Time-Series Overlay)
ax4 = plt.subplot(2, 2, 4)
ax4_2 = ax4.twinx() # 이중 y축 생성
# 위치 에너지 시계열
l1 = ax4.plot(t[100:1000], Ep_history[100:1000], color='blue', alpha=0.7, label='Potential Energy (Stress)')
# 방향 전환(Tumbling) 사건 시계열
l2 = ax4_2.plot(t[100:1000], p_tumble_history[100:1000], color='red', alpha=0.7, label='Tumble Probability')
ax4.set_title('Time-Series Dynamics (Sample Segment)', fontsize=12)
ax4.set_xlabel('Time')
ax4.set_ylabel('Potential Energy', color='blue')
ax4_2.set_ylabel('Tumble Probability', color='red')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='upper right')

plt.tight_layout(rect=[0, 0.03, 1,0.95])
plt.show()