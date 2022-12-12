import numpy as np
import matplotlib.pyplot as plt

history = np.load("history.npy", allow_pickle=True).item()
# summarize history for accuracy
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks(range(0, len(history["accuracy"])))
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xticks(range(0, len(history["loss"])))
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
plt.savefig('loss.png')
plt.show()