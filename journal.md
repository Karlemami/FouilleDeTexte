# 21 Mars

On a trouvé un corpus de musiques avec lyrics et genre <a href = "https://huggingface.co/datasets/Veucci/lyric-to-3genre">ici</a>.

On a tout de suite remarqué qu'il y avait des problèmes d'encodage au niveau des espaces : certains espaces étaient encodés par le caractère Unicode <a href="https://www.compart.com/en/unicode/U+2005">U+2005, "Four-Per-Em-Space</a>. On a remplacé tous ces caractères par l'espace classique (encodé 20), et ça a l'air d'avoir fonctionné.

## Avantages du corpus
Ce corpus est bien parce qu'il est pas trop volumineux (1000 chansons par genre), on va pouvoir vite tester des trucs dessus, et il a aussi l'avantage d'être sous licence libre.

## Inconvénients du corpus

Il ne contient que 3 genres (pop, rock et hip hop)

##
On a trouvé un autre corpus <a href="https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics">ici</a>, plus grand, qui contient 2,6 millions de chansons réparties en 6 genres. Une fois qu'on aura construit nos modèles, on pourra peut-être l'utiliser.