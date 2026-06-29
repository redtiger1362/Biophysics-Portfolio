#1. 고성능 연구 장비(라이브러리) 불러오기
import numpy as np
# 수정전(오류) import matplotlib as plt
# 수정후(정상) 
import matplotlib.pyplot as plt

#2. 감쇠진동 시스템의 물리적 상수 세팅
m= 1.0 # Mass
k = 10.0 # Spring constant
gamma = 0.5 # Damping Coefficient(세포질 점성저항)
A = 1.0 # 초기 진폭(위치)

#3. 시간 배열 만들기 (0~10초, 1000개 프레임)
t = np.linspace(0, 10, 1000)

#4. 각진동수(omega) 계산 및 감쇠진동수 수식 번역
omega = np.sqrt(k/m) # 각진동수
x = A * np.exp(-gamma*t/(2*m)) * np.cos(omega*t) # 감쇠진동수 수식

#5. 감쇠진동수 그래프 그리기(시각화)
plt.figure(figsize=(10, 5))  #도화지 크기 설정(가로 10, 세로5)
plt.plot(t, x, label='Damped Oscillation', color='blue')  # 시간과 위치 데이터를 이용하여 그래프 그리기

#6. 그래프 꾸미기
plt.title('Damped Harmonic Oscillation in Viscous Medium')  # 그래프 제목 설정
plt.xlabel('Time (t)')  # x축 라벨 설정
plt.ylabel('Displacement (x)')  # y축 라벨 설정
plt.axhline(0, color='black', lw=1)  # y=0 기준선 그리기
plt.grid(True)  # 격자 표시
plt.legend()  # 범례(label) 표시

#7. 화면에 출력
plt.show()  # 그래프 화면에 출력