import pandas as pd

df = pd.read_csv('cardio_train.csv', sep=';')

print('--- Dataset Size ---')
print(f"Sample Size (rows): {df.shape[0]}")
print(f"Feature Size (columns): {df.shape[1]}")

print('\n--- Extreme Values Check ---')
print(f"ap_hi min: {df['ap_hi'].min()}, max: {df['ap_hi'].max()}")
print(f"ap_lo min: {df['ap_lo'].min()}, max: {df['ap_lo'].max()}")
print(f"height min: {df['height'].min()}, max: {df['height'].max()}")
print(f"weight min: {df['weight'].min()}, max: {df['weight'].max()}")

print(f"\nNegative ap_hi: {(df['ap_hi'] < 0).sum()}")
print(f"Negative ap_lo: {(df['ap_lo'] < 0).sum()}")
print(f"ap_hi > 300: {(df['ap_hi'] > 300).sum()}")
print(f"ap_lo > 200: {(df['ap_lo'] > 200).sum()}")
print(f"height < 100: {(df['height'] < 100).sum()}")
print(f"height > 220: {(df['height'] > 220).sum()}")
print(f"weight < 30: {(df['weight'] < 30).sum()}")
print(f"weight > 200: {(df['weight'] > 200).sum()}")

print('\n--- Class Balance ---')
print(df['cardio'].value_counts())
print(f"Class ratio: {df['cardio'].value_counts()[1] / df['cardio'].value_counts()[0]:.2f}")

print('\n--- Feature Correlation with Target ---')
corr = df.corr()['cardio'].sort_values(ascending=False)
print(corr)
