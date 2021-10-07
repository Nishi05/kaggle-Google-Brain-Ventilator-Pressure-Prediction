# kaggle-Google-Brain-Ventilator-Pressure-Prediction
kaggle日記  
何を今やっているかのために整理する。  
データの中身  
・id：データのID  
・breath_id：呼吸毎のID  
・R：流量変化に対する圧力変化  
・C：圧力変化に対する体積変化  
・time_step：データ計測の時間（3秒まで）  
・u_in：吸気弁の開き具合（0～1の連続値）  
・u_out：呼気弁の開閉（0, 1の2値）  

## 2021-10-09(木)
今出ている特徴量を書いていく。  
参照リンク  
https://www.kaggle.com/manabendrarout/single-bi-lstm-model-pressure-predict-gpu-infer  

cumsum() 累積和
### time_stepとu_inの面積
df['area'] = df['time_step'] * df['u_in']
df['area'] = df.groupby('breath_id')['area'].cumsum()
### breath_id毎のu_inの累積和
df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
###  breath_id毎のu_inをずらして代入（shihtの数の分だけずらして行に追加する）
df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)  
df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)  
df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(3)  
df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(4)  
df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)  
df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-2)  
df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-3)  
df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-4)  
#### イメージ
![image](https://user-images.githubusercontent.com/40897913/136313441-ff329bfe-c6d9-4d07-8e70-91df497283d5.png)

### naを0埋め
df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
### breath_id毎のu_inの最大値
df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
### ずらしたlagとの差
df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
### brath_id毎の最大値との差
df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
### brath_id毎の平均値との差
df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
### RとCを文字列に変換
df['R'] = df['R'].astype(str)  
df['C'] = df['C'].astype(str)  
### RとCを合わせたものを文字列に変換
df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)



