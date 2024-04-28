from classifiers import *
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd

def load_csv(file_path: str):
    df = pd.read_csv(file_path,sep="|")
    lyrics = list(df["Lyrics"])
    genre = list(df["Genre"])
    return lyrics, genre
    

def vectorize(sentences: list[str]):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    return matrix

def generate_confusion_matrix(prediction, classes):
    labels = sorted(list(set(classes)))
    cm = confusion_matrix(classes, prediction, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    lyrics, classes = load_csv(args.file_path)
    matrix = vectorize(lyrics)
    
    predictions = random_forest(matrix, classes)
    generate_confusion_matrix(predictions, classes)
    print(classification_report(predictions,classes))
    
if __name__ == '__main__':
    main()
