import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # 엑셀 장비 판다스

m= 1.0
k = 10.0
A = 1.0
gamma_list = [0.1, 0.5, 2.0]
labels = ['water (gamma=0.1)', 'cytoplasm (gamma=0.5)', 'Dense fluid (gamma=2.0)']

t = np.linspace(0, 10, 1000)
omega = np.sqrt(k/m)

# 엑셀의 열이 될 빈 상자 만들기(첫 번째 열은 시간(Time)으로 설정)
data_dict = {'Time': t}

plt.figure(figsize=(10, 5))

for i in range(3):
    gamma = gamma_list[i]
    x = A * np.exp(-gamma*t/(2*m)) * np.cos(omega*t)
    plt.plot(t, x, label=labels[i])
    
    # 계산된 x 데이터를 엑셀의 열로 추가하기
    data_dict[labels[i]] = x

plt.title('Comparison of Damped Harmonic Oscillation in Different Viscous Mediums')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (x)')
plt.axhline(0, color='black', lw=1)
plt.legend()
plt.grid(True)

# 모은 데이터를 표로 만들고 CSV 파일로 저장하기
df = pd.DataFrame(data_dict) # 1. 상자에 담긴 데이터를 2차원 표(데이터프레임)로 변환
df.to_csv('damped_simulation.csv', index=False) # 2. 표를 CSV 파일 (엑셀 호환)로 저장
print("Data saved to 'damped_simulation.csv'")

plt.show()
