import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# --- KONFIGURACIJA ---
BATCH_SIZE = 32
IMG_SIZE = (128, 128) # Smanjujemo sa 512 na 128
EPOCHS = 10           # Dovoljno za prvi test
DATA_DIR = '../dataset'  # Putanja do foldera

print("Učitavam podatke...")

# Trening (Shuffle je UKLJUČEN)
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Validacija (Shuffle je ISKLJUČEN - bitno za konzistentnost)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'val'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

# Test (Shuffle je ISKLJUČEN - OBAVEZNO da bismo povezali sliku sa greškom)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

class_names = train_ds.class_names
print(f"Klase: {class_names}")
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1), # Rotacija do 10% (blago)
    layers.RandomZoom(0.2),     # Zoom do 20% (KLJUČNO za vuka vs psa!)
    layers.RandomContrast(0.1), # Malo menjamo kontrast
])


model = models.Sequential([
    # Ulazni sloj + Varijacija slika + Normalizacija (0-255 -> 0-1)
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    
    # Blok 1
    layers.Conv2D(16, (3, 3), activation='relu'), # Manji broj filtera za početak
    layers.MaxPooling2D((2, 2)),
    
    # Blok 2
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Blok 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Klasifikator
    layers.Flatten(),
    layers.Dense(64, activation='relu'), # Jedan skriveni sloj

    layers.Dropout(0.5),
    
    layers.Dense(3, activation='softmax')
])

# Kompilacija
model.compile(optimizer='adam', # Standardni LR je 0.001
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("\nPočinjem trening...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',          # Pratimo validacionu grešku
    patience=5,                  # Čekamo 5 epoha ako nema poboljšanja
    restore_best_weights=True,   # Vraćamo model na stanje sa najmanjom greškom
    verbose=1                    # Ispisuje poruku kad prekine trening
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# --- 4. EVALUACIJA I GRAFICI ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# --- 5. ANALIZA GREŠAKA NA TEST SETU ---
print("\nGenerišem predviđanja za test set...")

# Izvlačimo slike i labele iz test seta u memoriju (ovo radi brzo za 2-3k slika)
test_images = []
test_labels = []
for images, labels in test_ds:
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# Predviđanje
predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)

# Pronalaženje grešaka
errors = np.where(pred_labels != test_labels)[0]
print(f"\nUkupno grešaka: {len(errors)} od {len(test_labels)} slika.")
print(f"Tačnost na testu: {(1 - len(errors)/len(test_labels))*100:.2f}%")

# Prikaz 10 najgorih grešaka (gde je model bio najsigurniji u pogrešnu odluku)
# Sortiramo greške po "poverenju" modela (najveća verovatnoća pogrešne klase)
error_confidences = []
for i in errors:
    confidence = predictions[i][pred_labels[i]] # Koliko je bio siguran u grešku
    error_confidences.append((confidence, i))

# Sortiraj opadajuće - želimo da vidimo gde je model bio "ubeđen" da je u pravu, a grešio je
error_confidences.sort(key=lambda x: x[0], reverse=True)

print("\n--- TOP 10 'NAJGLUPLJIH' GREŠAKA (Model bio siguran, a pogrešio) ---")
plt.figure(figsize=(15, 8))
for i, (conf, idx) in enumerate(error_confidences[:10]): # Prikazujemo top 10
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx].astype("uint8"))
    
    true_class = class_names[test_labels[idx]]
    pred_class = class_names[pred_labels[idx]]
    
    plt.title(f"T: {true_class}\nP: {pred_class}\nConf: {conf:.2f}")
    plt.axis("off")
plt.tight_layout()
plt.show()