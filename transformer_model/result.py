import numpy as np
def classification_report(cfs_matrix):
    TN, FP, FN, TP = cfs_matrix.ravel()
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    return Recall, Precision, F1_score
    


def save_plot(train_history,val_history,title,EPOCHS,ax):
    points = EPOCHS
    x =  (range(points+1))[1:]
    ax.plot(x, train_history,label = "train")
    ax.plot(x, val_history,label = "val")
    ax.set_title(title)
    return

def record_point(writer,epoch,train_acc,val_acc,train_loss,val_loss,train_F1_score,val_F1_score):
    writer.add_scalars("accuracy",{
    'train': train_acc.item(),
    'validation': val_acc.item(),
    }, epoch+1)
    writer.add_scalars("loss ",{
    'train': np.asscalar(train_loss),
    'validation': np.asscalar(val_loss),
    }, epoch+1)
    writer.add_scalars("F1_score ",{
    'train': np.asscalar(train_F1_score),
    'validation': np.asscalar(val_F1_score),
    }, epoch+1)