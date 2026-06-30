import numpy as np
import matplotlib.pyplot as plt


# 1. 시뮬레이션 환경 설정
n_steps = 1000 # 입자가 이동할 총 횟수 (시간)

# 2. [핵심] 무작위 스탭 생성 (동전 던지기)
# np.random.choice는 [-1,1] 중에서 하나를 50% 확률로 무작위로 뽑음
steps = np.random.choice([-1,1], size=n_steps)

# 3. [핵심] 입자의 현재 위치 계산 (누적)
# np.cumsum은 지금까지 걸어온 스탭들을 계속 누적해서 현재 위치 추적
# 입자가 매 순간 움직인 스텝(step_x) 데이터가 다음과 같이 나왔다고 가정하면,
# step_x = [1.5, -0.5, 2.0, -1.0] (매 순간의 발걸음)
# 이 배열을 np.cumsum() 하면 다음과 같이 누적해서 더해진 배열이 탄생
# x = [1.5, (1.5 - 0.5), (1.0 + 2.0), (3.0 - 1.0)]

position = np.cumsum(steps)

# 4. 결과 시각화
plt.figure(figsize=(10,5))
plt.plot(position, color='purple', alpha=0.8, lw= 1.5)
# 가로축 데이터(t)를 주지 않아도 가로축에 n_steps가 나온 것은 Matplotlib특성
# 배열 하나만 입력하면 자동으로 가로축을 그 배열의 인덱스 번호로 매핑

# 그래프 꾸미기
plt.title("1D random walk Simulation")
plt.xlabel("Time (Number of steps)")
plt.ylabel("Position (x)")
plt.axhline(0, color='black', linestyle='--')
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()