import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Podešavanje stila grafika
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Definisanje putanje
dataset_dir = '../dataset'
subsets = ['train', 'val', 'test']

# 2. Prikupljanje podataka
data = []

# Prolazimo kroz train, val i test
for subset in subsets:
    subset_path = os.path.join(dataset_dir, subset)
    if not os.path.exists(subset_path):
        continue
        
    # Prolazimo kroz klase (dog, cat, wild)
    for class_name in os.listdir(subset_path):
        class_path = os.path.join(subset_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            data.append({
                'Set': subset,
                'Klasa': class_name,
                'Broj slika': count
            })

# Kreiranje Pandas tabele
df = pd.DataFrame(data)

# 3. Prikaz tabele
print("Detaljna statistika po setovima:")
pivot_table = df.pivot(index='Klasa', columns='Set', values='Broj slika')
# Dodajemo kolonu Ukupno
pivot_table['Ukupno'] = pivot_table.sum(axis=1)
pivot_table = pivot_table[['train', 'val', 'test', 'Ukupno']]
print(pivot_table)
print("-" * 50)

# Provera balansa (izračunavanje razlike)
max_imgs = pivot_table['Ukupno'].max()
min_imgs = pivot_table['Ukupno'].min()
diff_percent = ((max_imgs - min_imgs) / max_imgs) * 100

print(f"\nNajbrojnija klasa ima: {max_imgs} slika")
print(f"Najmanja klasa ima: {min_imgs} slika")
print(f"Razlika u balansu: {diff_percent:.2f}%")

if diff_percent < 15:
    print("ZAKLJUČAK: Podaci SU balansirani.")
else:
    print("ZAKLJUČAK: Podaci NISU balansirani.")

# 4. Crtanje Grafikona
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Graf 1: Ukupan broj slika po klasama
sns.barplot(data=df.groupby('Klasa')['Broj slika'].sum().reset_index(), 
            x='Klasa', y='Broj slika', hue='Klasa', ax=ax[0], palette='viridis')
ax[0].set_title('Ukupna distribucija slika po klasama', fontsize=14)
ax[0].set_ylabel('Ukupan broj slika')
ax[0].bar_label(ax[0].containers[0]) # Dodaje brojke na stubiće

# Graf 2: Raspodela po Train/Val/Test
pivot_table[['train', 'val', 'test']].plot(kind='bar', stacked=True, ax=ax[1], color=['#3498db', '#f1c40f', '#e74c3c'])
ax[1].set_title('Raspodela podataka po podskupovima (Train/Val/Test)', fontsize=14)
ax[1].set_ylabel('Broj slika')
ax[1].legend(title='Podskup')

plt.tight_layout()
plt.show()