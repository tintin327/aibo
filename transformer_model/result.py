import numpy as np
import sqlite3

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
    
def record_predicted_of_url(writer,predicted_of_url):
    tp,tn,fp,fn = predicted_of_url
    conn = sqlite3.connect('aibo.db')
    cursor = conn.execute("SELECT * FROM parsed_news")
    parsed_news = cursor.fetchall() 
    articles = []
    
    cursor = conn.execute("SELECT * FROM parsed_news")
    DATA = cursor.fetchall()

    for neg in fn:
        for parsed in parsed_news:
            if neg == parsed[1]:
                article_text = parsed[3]
                writer.add_text('False Negative url', neg)
                writer.add_text('False Negative article', article_text)
    for pos in fp:
        for parsed in parsed_news:
            if pos == parsed[1]:
                article_text = parsed[3]
                writer.add_text('False Positive url', pos)
                writer.add_text('False Positive article', article_text)
                
def save_predicted_file(predicted_of_url):
    conn = sqlite3.connect('aibo.db')
    cursor = conn.execute("SELECT * FROM parsed_news")
    parsed_news = cursor.fetchall() 
    tp,tn,fp,fn = predicted_of_url
    conn = sqlite3.connect('result_of_model.db')
                    
    print ("Opened database successfully");
    conn.execute('''DROP TABLE true_positive''');
    conn.execute('''DROP TABLE false_positive''');
    conn.execute('''DROP TABLE true_negative''');
    conn.execute('''DROP TABLE false_negative''');
    conn.commit()

    conn.execute('''CREATE TABLE true_positive
           (ID INT PRIMARY KEY     NOT NULL,
           URL           TEXT    NOT NULL,
           ARTICLE        TEXT );''')
    conn.execute('''CREATE TABLE false_positive
           (ID INT PRIMARY KEY     NOT NULL,
           URL           TEXT    NOT NULL,
           ARTICLE        TEXT );''')
    conn.execute('''CREATE TABLE true_negative
       (ID INT PRIMARY KEY     NOT NULL,
       URL           TEXT    NOT NULL,
       ARTICLE        TEXT );''')
    conn.execute('''CREATE TABLE false_negative
       (ID INT PRIMARY KEY     NOT NULL,
       URL           TEXT    NOT NULL,
       ARTICLE        TEXT );''')
    
    print ("Table created successfully");

    conn.commit()

    tp_num = 0
    fp_num = 0
    tn_num = 0
    fn_num = 0
    
    for parsed in parsed_news:
        for tpos in tp:
            if tpos == parsed[1]:
                tp_num = tp_num+1
                article_text = parsed[3]
                conn.execute("insert into true_positive (ID,URL,ARTICLE) values (?, ?, ?)",(tp_num,  tpos, article_text))
        for tneg in tn:
            if tneg == parsed[1]:
                tn_num = tn_num+1
                article_text = parsed[3]
                conn.execute("insert into true_negative (ID,URL,ARTICLE) values (?, ?, ?)",(tn_num,  tneg, article_text))
        for fpos in fp:
            if fpos == parsed[1]:
                fp_num = fp_num+1
                article_text = parsed[3]   
                conn.execute("insert into false_positive (ID,URL,ARTICLE) values (?, ?, ?)",(fp_num,  fpos, article_text))
        for fneg in fn:
            if fneg == parsed[1]:
                fn_num = fn_num+1
                article_text = parsed[3]
                conn.execute("insert into false_negative (ID,URL,ARTICLE) values (?, ?, ?)",(fn_num,  fneg, article_text))

    conn.commit()
    print ("Records Insert successfully");

    conn.close()


    
    
    
    
    
    
    
    
    