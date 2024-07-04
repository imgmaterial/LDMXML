
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, np.flip(target_names,axis=0))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Counted number of electrons')
    plt.xlabel('Simulated number of electrons')
    plt.show()
    
def make_cm_plot(model,inp,
            trg,
            num_classes,
            label='Test data'):
    print("Predicting")
    y = model.predict(inp, verbose=1, batch_size = 32 )
    print("Post prediction")
    #d = [trg[i][1][x] for i in range(len(trg)) for x in range(len(trg[i][1]))]
    d = [trg[i][1] for i in range(len(trg))]
    print("cool")
    d = np.concatenate(d, axis = 0)
    print(d)
    d_class = d.argmax(axis=1)
    y_class = y.argmax(axis=1)
    print('accuracy:   ', '{:.4f}'.format((y_class==d_class).mean()), '\n')

    class_names = ['{}'.format(i+1) for i in range(num_classes)]
    #print(classification_report(d_class, y_class, target_names=class_names))

    confuTst = np.flip(confusion_matrix(d_class, y_class,).T,axis=0)

    plot_confusion_matrix(cm           = confuTst, 
                          normalize    = False,
                          target_names = class_names,
                          title        = "Confusion Matrix")
