Dépôt pour le projet du cours de Fouille de Textes (2023_2024) - Master PluriTAL  

## Projet
**Analyse comparative des performances de différents algorithmes de classification**  
› Ce projet vise à réaliser une analyse comparative des performances de plusieurs algorithmes de classification en machine learning, notamment Naïve Bayes, Support Vector Machine (SVM), Decision Tree et Random Forest. L'objectif est de comparer leur efficacité en termes de précision et d'autres critères de performance en fonction de différentes données.

## Corpus 
Les corpus ont été trouvés sur HuggingFace.  

**Premier corpus** : 'Veucci/lyric-to-3genre'
  - 3000 chansons en anglais
  - 3 classes : Pop, Rock, Hip-hop (répartition équitable entre les classes)

  **Second corpus** : 'juliensimon/autonlp-data-song-lyrics-demo'
  - 54 000 chansons en anglais
  - 6 classes : Rock, Indie, Hip-Hop, Pop, Heavy Metal et Dance (répartition inégale)


## Pré-traitement
**Premier corpus** : 'Veucci/lyric-to-3genre'  
Deux prétraitements différents pour avoir deux "nouveaux" corpus : 
  - un corpus lemmatisé                      
  - un corpus racinisé

  Comparaison de l'efficacité des modèles selon les deux prétraitements.  

 **Second corpus** : 'juliensimon/autonlp-data-song-lyrics-demo'  
   - équilibrage des classes avec *pandas*

## Méthode 
**Second corpus** : 'juliensimon/autonlp-data-song-lyrics-demo'  
Nous cherchons à établir si l’efficacité d’un modèle de classification est liée à la similarité cosinus entre deux genres musicaux.
1. création de paires de genre uniques (15 paires).
2. concaténation par classe de toutes les chansons en une seule chaîne de caractères afin de vectoriser tout le corpus en un seul espace vectoriel commun.
3. ajustez le vectoriseur sur l’ensemble des mots de toutes les classes confondues pour avoir le même nombre de dimensions.
4. transformez chaque document de classe unique en vecteur à l'aide du vectoriseur ajusté.
5. calculer l'exactitude et la similarité cosinus

## Résultats
**Premier corpus** : 'lyric-to-3genre'  
Résultats satisfaisants, en particulier le SVM et le random forest.  
La classe “Pop” a été systématiquement la moins bien classée, souvent confondue avec la classe “Rock”.  
La classe “Hip-Hop” a obtenu les meilleures performances, probablement en raison de ses traits discriminants plus prononcés.

**Second corpus** : 'juliensimon/autonlp-data-song-lyrics-demo'  
Les algorithmes de classification n'ont pas dépassé un f-score de 0.45, ce qui souligne la complexité d'un problème de classification à six classes.  
Analyse de la similarité cosinus -> résultat notable : similarité élevée entre les genres Dance et Heavy Metal (cosinus de 0.94).
L'efficacité d’un modèle semble être corrélé au degré de similarité entre les classes, la présence de quelques attributs très discriminants joue sans doute également un rôle important.

## Conclusion 
Les paroles seules semblent insuffisantes pour une classification précise des genres musicaux. D'autres attributs, comme le tempo ou les caractéristiques instrumentales, pourraient être également discriminants.
Une approche multimodale, combinant l'analyse des paroles et des données audio, pourrait améliorer significativement la robustesse et la précision des modèles de classification.
