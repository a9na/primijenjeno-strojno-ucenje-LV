import pandas as pd
import matplotlib.pyplot as plt

mtcars = pd.read_csv('mtcars.csv')

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].bar(['4 cilindra', '6 cilindara', '8 cilindara'], mtcars.groupby('cyl')['mpg'].mean(), color=['blue', 'green', 'red'])
axs[0, 0].set_title('Prosječna potrošnja goriva po broju cilindara')
axs[0, 0].set_xlabel('Broj cilindara')
axs[0, 0].set_ylabel('Prosječna potrošnja goriva (mpg)')
axs[0, 0].grid(True)

mtcars.boxplot(column='wt', by='cyl', ax=axs[0, 1])
axs[0, 1].set_title('Distribucija težine po broju cilindara')
axs[0, 1].set_xlabel('Broj cilindara')
axs[0, 1].set_ylabel('Težina (tons)')
axs[0, 1].grid(True)

mtcars.groupby('am')['mpg'].mean().plot(kind='bar', color=['blue', 'red'], ax=axs[1, 0])
axs[1, 0].set_title('Prosječna potrošnja goriva za automobile s ručnim i automatskim mjenjačem')
axs[1, 0].set_xlabel('Tip mjenjača')
axs[1, 0].set_ylabel('Prosječna potrošnja goriva (mpg)')
axs[1, 0].set_xticks([0, 1], ['Ručni', 'Automatski'])
axs[1, 0].grid(True)

axs[1, 1].scatter(mtcars[mtcars['am'] == 1]['hp'], mtcars[mtcars['am'] == 1]['qsec'], color='blue', label='Automatski')
axs[1, 1].scatter(mtcars[mtcars['am'] == 0]['hp'], mtcars[mtcars['am'] == 0]['qsec'], color='red', label='Ručni')
axs[1, 1].set_title('Odnos ubrzanja i snage za automobile s ručnim i automatskim mjenjačem')
axs[1, 1].set_xlabel('Snaga (hp)')
axs[1, 1].set_ylabel('Ubrzanje (s)')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
