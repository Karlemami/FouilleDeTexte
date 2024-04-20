import stanza
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


class Lemmatizer:
    def __init__(self):
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma")
    
    def lemmatize(self, text):
        doc = self.nlp(text)
        lemmas = [
            word.lemma
            for sentence in doc.sentences
            for word in sentence.words
        ]
        return " ".join(lemmas)


class Stemmer:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
    
    def stem(self,text):
        tokens = word_tokenize(text)
        roots = [self.stemmer.stem(token)for token in tokens]
        return " ".join(roots)

def main():
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus",type=str)
    parser.add_argument("output_file",type=str)
    parser.add_argument("-p","--processes",choices=["lemmatizer","stemmer"],nargs="+")
    args = parser.parse_args()
    
    df = pd.read_csv(args.corpus,sep="|",header=None)
    df.columns = ["Genre","Lyrics"]
    
    if "lemmatizer" in args.processes:
        l = Lemmatizer()
        df["Lyrics"] = df["Lyrics"].apply(Lemmatizer.lemmatize())
    if "stemmer" in args.processes:
        s = Stemmer()
        df["Lyrics"] = df["Lyrics"].apply(Stemmer.stem())
    
    df.to_csv(args.output_file,index=False,sep="|")
    
if __name__ == "__main__":
    main()