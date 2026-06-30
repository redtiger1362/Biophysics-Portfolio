import numpy as np
import matplotlib.pyplot as plt

# 1. 미생물 및 환경 파라미터
n_steps = 1000
v_run = 1.0 # 편모 모터를 이용한 직진 속도
p_tumble = 0.1  # 매 순간 방향을 무작위로 바꿀 확률
B_field_bias = 2.0  # 자기장 편향 강도, 값이 크면 북쪽 지향, 0이면 무작위

# 2. 상태 저장소 초기화
x,y = np.zeros(n_steps), np.zeros(n_steps)
theta = np.random.uniform(0,2 * np.pi)  # 0~360도 사이 초기 출발 방향
# np.random.uniform(0,2 * np.pi)은 [최솟값,최댓값) 사이에서 균등분포 난수 생성

# 3. [핵심] Biased Run-and-Tumble loop
for i in range(1, n_steps):
    #난수를 뽑아 Tumble 확률(10%)에 해당하는지 확인
    if np.random.rand() < p_tumble:
        # np.random.rand()는 [0.0,1.0) 구간에서 균등분포를 따르는 실수 난수 1개 무작위 생성
        # Tumble 발생: 방향(theta) 재설정
        theta=np.random.vonmises(mu=np.pi/2, kappa=B_field_bias)
        # vonmises(mu, kappa)는 원형 정규분포 모델
        # mu는 중심각도, kappa는 중심각도에 편향되게 난수 생성하며 클수록 집중,0이면 완전균등

        # 현재 바라보고 있는 각도 방향으로 v_run만큼 전진
        # 삼각함수를 이용해 2차원 평면의 x,y 이동량 계산
    x[i] = x[i-1] + v_run * np.cos(theta)
    y[i] = y[i-1] + v_run * np.sin(theta)

# 4. 고해상도 궤적 시각화
plt.figure(figsize=(8,10))
plt.plot(x,y, color='teal', lw=1.2, alpha=0.8, label='Bacterial Trajectory')

# 양 끝값 표시
# s는 마커의 면적, marker는 시각적 형태로 '*' 는 별, 'x'는 X 모양
plt.scatter(x[0], y[0], color='black', s=100, label='start', zorder=5)  
plt.scatter(x[-1], y[-1], color='red',marker='*', s=150, label='End (North Bound)', zorder=5)

# 자기장 방향 추
plt.axvline(x[0], color='gray', ls= '--', alpha=0.5)
plt.annotate('magnetic north', xy=(x[0],y[-1]), xytext=(x[0]-20, y[-1]+20),
             arrowprops=dict(facecolor='red', shrink= 0.05))
            # plt.annotate는 화살표 지시, xy는 끝점 좌표, xytext는 text 배치 좌표
            # arrowprops=dict()는 화살표 스타일, facecolor는 내부 색, shrink는 화살표 정밀 시각화

plt.title("Magnetotactic Bacterium: Biased Run-and-Tumble")
plt.xlabel("Position (x)")
plt.ylabel("Position (y)")
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.axis('equal')

plt.show()