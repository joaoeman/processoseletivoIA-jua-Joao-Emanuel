import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
 
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, stratify=y_train, test_size=0.25
)
 

# 2) Formatando o dataset para entrada da CNN 

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
 
input_shape = (28, 28, 1)
 
# Normalizando imagens para [0, 1] 
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
 

# 3) Construcao da CNN
#    Requisito: 3 blocos Conv2D + BatchNormalization + MaxPooling2D
#    seguidos de Dropout antes da saida

model = Sequential()
 
# Bloco 1
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
 
# Bloco 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
 
# Bloco 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
model.summary()
 

# 4) Compilacao (Adam agora importado corretamente)

adamOptimizer = Adam(learning_rate=0.001)
 
model.compile(
    optimizer=adamOptimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
 

# 5) Treinamento com EarlyStopping monitorando val_loss
#    epochs sobe para 15 porque o EarlyStopping decide quando parar de verdade;

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
 
history = model.fit(
    x=x_train, y=y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=16,
    shuffle=False,
    callbacks=[early_stop]
)

print(f"Treinamento parou na epoca: {len(history.epoch)}")  
 
# 6) Exibicao explicita da acurácia de validacao final 
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\nAcuracia de validacao final: {val_acc:.4f}")
print(f"Perda de validacao final:    {val_loss:.4f}")
 

# 7) Graficos salvos em arquivo 

pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('Acuracia do modelo no treino e validacao')
pyplot.ylabel('Acuracia')
pyplot.xlabel('Epoca')
pyplot.legend(['Treino', 'Validacao'], loc='upper left')
pyplot.savefig('historico_acuracia.png')
pyplot.close()
 
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Perda do modelo no treino e validacao')
pyplot.ylabel('Perda')
pyplot.xlabel('Epoca')
pyplot.legend(['Treino', 'Validacao'], loc='upper left')
pyplot.savefig('historico_perda.png')
pyplot.close()
 
# 8) Salvamento do modelo treinado 
model.save("model.h5")
print("Modelo salvo em model.h5")
 
