import scipy.io
import pandas as pd

gen_1 = scipy.io.loadmat('./data/mat_data/gen_1_data.mat')
gen_2 = scipy.io.loadmat('./data/mat_data/gen_2_data.mat')
gen_3 = scipy.io.loadmat('./data/mat_data/gen_3_data.mat')

real_1 = gen_1['usol1']
real_2 = gen_2['usol2']
real_3 = gen_3['usol3']

df_1 = pd.DataFrame(real_1)
df_2 = pd.DataFrame(real_2)
df_3 = pd.DataFrame(real_3)

# 保存到 Excel 文件
df_1.to_excel('gen1_real.xlsx', index=False)
df_2.to_excel('gen2_real.xlsx', index=False)
df_3.to_excel('gen3_real.xlsx', index=False)