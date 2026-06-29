import numpy as np
import matplotlib.pyplot as plt

# 1. 고정된 물리 상수
m = 1.0 
k = 10.0
A = 1.0

# 2. 비교할 3가지 환경의 감쇠 계수(gamma) 리스트(물,세포질, 매우 끈적한 유체)
gamma_list = [0.1, 0.5, 2.0] 
labels = ['water (gamma=0.1)', 'cytoplasm (gamma=0.5)', 'Dense fluid (gamma=2.0)']

# 3. 시간 배열 및 각진동수 계산
t = np.linspace(0, 10, 1000)
omega = np.sqrt(k/m)

# 4. 도화지 세팅
plt.figure(figsize=(10, 5))

# 5. [핵심] for 반복문을 사용하여 3개의 수식을 한 번에 계산하고 그리기
for i in range(3):
    gamma = gamma_list[i]
    x = A * np.exp(-gamma*t/(2*m)) * np.cos(omega*t)  # 감쇠진동수 수식
    plt.plot(t, x, label=labels[i])  # 그래프 그리기

# 6. 그래프 꾸미기
plt.title('comparison of Damped Harmonic Oscillation in Different Viscous Mediums')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (x)')
plt.axhline(0, color='black', lw=1)
plt.grid(True)
plt.legend()

plt.show()  # 그래프 화면에 출력