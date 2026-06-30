import numpy as np
import matplotlib.pyplot as plt

# 1. 시뮬레이션 환경 설정
n_steps = 10000  # 입자가 이동할 횟수 (1차원보다 길게)

# 2. [핵심] 2차원 무작위 스텝 생성 (Gaussian 정규 분포)
# 실제 무리 세계에서 세포질 내 분자들의 충돌은 동전 던지기가 아님
# 평균이 0이고 표준 편차가 1인 Normal Distribution 난수를 사용
step_x = np.random.normal(loc=0.0, scale=1.0, size= n_steps)
step_y = np.random.normal(loc=0.0, scale=1.0, size= n_steps)
# loc는 확률 분포의 중심, loc이 양수이면 입자도 전체적으로 오른쪽으로 움직이는 경향을 보임
# scale 은 표준 펀차로 입자가 한번에 움직이는 평균적인 걸음 크기
# size는 무작쉬 숫자를 몇 개 생성할 것인지 지정

# 3. 2차원 평면에서의 위치 계산(누적합)
x = np.cumsum(step_x)
y = np.cumsum(step_y)
# 같은 순서에 있는 x,y를 짝지어 평면 위에 점

# 4. 2차원 궤적 시각화
plt.figure(figsize=(8, 8))
plt.plot(x,y, color='royalblue', alpha=0.6, lw=0.5)

# 시작점과 끝점 강조 표시
# plt.scatter는 좌표 위치에 독립적인 점을 찍음
# zorder는 레이어 우선순위로 값이 클수록 위쪽 배치
plt.scatter(x[0],y[0], color='green', s=100, label='start', zorder=5)
plt.scatter(x[-1],y[-1], color='red', s=100, label='end', zorder=5)

# 그래프 꾸미기
plt.title("2D Brownian Motion Simultian (Single Particle)")
plt.xlabel("Position(x)")
plt.ylabel("Position(y)")
plt.grid(True, ls=':', alpha=0.7)
plt.legend()
plt.axis('equal')  # x,y축의 시각적 스케일 동일하게 강제, 궤적 왜곡 방지

plt.show()
