from classifiers import *
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_csv(file_path: str):
    classes = []
    lyrics = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file,delimiter='|')
        for row in reader:
            classes.append(row[0])
            lyrics.append(row[1])
    
    return lyrics, classes

def vectorize(sentences: list[str]):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    return matrix

def generate_confusion_matrix(prediction, classes):
    cm = confusion_matrix(classes, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pop', 'Rock', 'Hip-Hop'])
    disp.plot()
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    lyrics, classes = load_csv(args.file_path)
    matrix = vectorize(lyrics)
    
    nb_predictions = svm(matrix, classes)
    generate_confusion_matrix(nb_predictions, classes)
    
if __name__ == '__main__':
    main()
