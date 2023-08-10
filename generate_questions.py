# Complete required imports:
import pandas as pd 
import numpy as np

# Import data to generate questions from:
data_df = pd.read_csv('class_to_instruct.csv', sep=',')
instructs = list(data_df.columns)
instructs = instructs[1:]
num_instructs = len(instructs)
classes = list(data_df.values[:, 0])
num_classes = len(classes)
data = data_df.values[:, 1:]
num_questions = 50
questions = []
ids = []
count = 0

while count < num_questions:
    row = np.random.randint(num_classes - 1)
    col = np.random.randint(num_instructs - 1)
    if data[row][col] == 1:
        questions.append(str(instructs[col]) + " " + str(classes[row]) + ".")
        ids.append(str(col) + " " + str(row))
        count += 1

questions = np.array(questions)
np.savetxt("test.csv", questions, delimiter =',', fmt ='%s')
    