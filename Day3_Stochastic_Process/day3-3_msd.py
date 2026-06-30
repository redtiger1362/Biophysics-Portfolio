#입자가 원점으로부터 멀어진 거리의 '제곱'을 구한 뒤, 수많은 입자들의 평균을 내보면
# 그 값(MSD)이 시간($t$)에 완벽하게 정비례하며 직선으로 증가
# MSD(t) =  <r(t)**2> = 4Dt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 시뮬레이션 환경 설정
n_particles = 500  # 입자의 개수
n_steps =  1000

# 2. [핵심] 500개 입자의 무작위 스탭 동시 생성 (행렬)
# size=(500, 1000)으로 설정하여 500개 입자가 1000번 움직이는 2차원 배열(데이터 표) 생성
step_x = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, n_steps))
step_y = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, n_steps))

# 3. 누적합으로 각 입자의 위치 계산
# axis=1은 데이터 표에서 가로 방향(시간 흐름)으로 누적해서 더하라는 뜻
x= np.cumsum(step_x, axis=1)
y= np.cumsum(step_y, axis=1)

# 4. [핵심] 평균 제곱 변위(MSD) 계산
# 원점으로부터의 거리의 제곱
r_squared = x**2 + y**2

# 입자 500개의 r^2값을 각 스탭마다 평균내기
# axis=0은 데이터 표에서 세로 방향(500개 입자)로 평균을 구하라는 뜻
msd = np.mean(r_squared, axis=0)

# 5. 시간 축 생성 및 시각화
# np.arange에서 기본값 생략 가능 -> 시작점 없으면 0, 간격 없으면 1칸씩, n_steps에 따라 끝점은 999
time_steps = np.arange(n_steps)

plt.figure(figsize=(10,5))
plt.plot(time_steps, msd, color='firebrick', lw=2, label='Simulated MSD (500 particles)')

# 이론적 추세선 추가
# scale=1.0인 2차원 브라운 운동의 MSD는 시간에 비례하여 증가하므로 기울기 약 2로 잡음
plt.plot(time_steps, time_steps *2, color='black', ls='--', label='Theoretical Linear Trend')

plt.title("Mean Squared Displacement (MSD) of 500 Particles")
plt.xlabel("Time (t)")
plt.ylabel("MSD")
plt.legend()
plt.grid(True, ls=':', alpha=0.7)

plt.show()

# 6. CSV 추출, 시뮬레이션과 이론적으로 얻은 MSD 나란히 비교하는 표 생성
df_msd = pd.DataFrame({
    'Time_Step': time_steps,
    'Simulated_MSD': msd,
    'Theoretical_MSD': time_steps * 2.0
})

csv_filename= 'brownian_msd_data.csv'
df_msd.to_csv(csv_filename, index=False)
print(f"통계 분석 데이터 '{csv_filename}'로 추출 완료")
# print(f"...")는 format으로 문자열 내부에 단순 텍스트 외에 계산할 코드가 있음을 나타냄(f-string mode)
# ''안의 {}만 뚫어서 읽음