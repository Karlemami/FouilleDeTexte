import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from main import vectorize
from classifiers import *

def split_corpus_into_genres_matrices(input_file: str) -> dict:
    """On récupère tous les genres du corpus et on créé un nouveua dataframe par genre. On stocke les dataframes dans un dictionnaire.
        Les clés sont le nom du genre. Ça va nous permettre de faire les calculs pour chaque paire de genre."""
    df = pd.read_csv(input_file, sep="|")
    genres = list(set(df["Genre"]))
    genre_dataframes = {}
    for genre in genres:
        genre_dataframes[genre] = df[df["Genre"] == genre]
    return genre_dataframes


def get_cosine_similarity(df1, df2)->float:
    """Calcule la similarité cosinus entre deux genres. On récupère les lyrics en deux strings, un par genre, car on veut la similarité
    globale du corpus, et pas les similarités chanson par chanson."""
    s1 = ' '.join(list(df1["Lyrics"]))
    s2 = ' '.join(list(df2["Lyrics"]))
    vectorizer = TfidfVectorizer()
    vectorizer.fit([s1, s2]) # Le vectorizer doit avoir le même vocabulaire pour les deux corpus, sinon on peut pas comparer.
    m1 = vectorizer.transform([s1])
    m2 = vectorizer.transform([s2])
    similarity = cosine_similarity(m1,m2)[0][0] #renvoie une np.array similarité document par document. On a que deux documents : les deux corpus qu'on a converti en string
    return similarity
    
def main():
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    genre_matrices = split_corpus_into_genres_matrices(args.file_path)
    similarity_and_report = {}
    for genre1 in genre_matrices:
        for genre2 in genre_matrices: # On fait les calculs pour chaque paire de genre
            if genre1 != genre2:
                size_genre1 = len(genre_matrices[genre1]) #On veut que les deux corpus aient la même taille.
                size_genre2 = len(genre_matrices[genre2])
                if size_genre1 > size_genre2:
                    m1 = genre_matrices[genre1].sample(size_genre2)
                    m2 = genre_matrices[genre2]
                elif size_genre2 > size_genre1:
                    m1 = genre_matrices[genre1]
                    m2 = genre_matrices[genre2].sample(size_genre1)
                else:
                    m1 = genre_matrices[genre1]
                    m2 = genre_matrices[genre2]
                similarity = get_cosine_similarity(m1,m2)
                concat = pd.concat([m1,m2])
                lyrics = list(concat["Lyrics"])
                classes = list(concat["Genre"])
                matrix = vectorize(lyrics)
                predictions = random_forest(matrix, classes)
                classification = classification_report(predictions, classes, output_dict=True)
                accuracy = classification["accuracy"]
                similarity_and_report[(genre1, genre2)] = (similarity, accuracy)
    
    for pair in similarity_and_report:
        print(f"{pair[0]}--{pair[1]}: similarity : {similarity_and_report[pair][0]}     accuracy : {similarity_and_report[pair][1]} ")
    
    similarities = []
    accuracies = []
    for pair in similarity_and_report:
        similarities.append(similarity_and_report[pair][0])
        accuracies.append(similarity_and_report[pair][1])

    plt.scatter(similarities, accuracies)
    plt.xlabel('Similarity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Cosine Similarity')
    plt.show()

    
if __name__ == '__main__':
    main()
    