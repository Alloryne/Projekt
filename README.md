Entropy cluster.
https://entropy-doc.mimuw.edu.pl/registeringusers.html

Articles:
https://arxiv.org/pdf/1706.03459
Selecting a winner with external referees

Notebook:
https://colab.research.google.com/drive/1XNDCkFNG3KNA9l9b1AuUKcAfkDmmmbp_

Project text:
https://www.overleaf.com/project/68fe728d5787c7dcef1d77bc

Ideas:
Ideas for score misreporting:
1. Error of reporting means that players score gets reported as $1 - s$, where $s$ is the actual score. The score gets misreported with some set probability smaller than $\frac{1}{2}$.
2. Error of reporting means that player's score gets reported as $s + \frac{1}{2} \mod 1$, where $s$ is the actual score. The score gets misreported with some set probability smaller than $\frac{1}{2}$.
3. Reported scores are actual scores with random error term from set distribution added.
4. Reported scores are actual scores with random error term from set distribution added and then normalized back into correct range.

Situations:
1. $n$ players. Each player's work has a score from distribution $F$, that is known to the player. The planner has to choose one of the players based on the scores that they report and additional independent reports that can be erroneous.
2. $n$ players. Each player's work has a score from distribution $F$. Players know their own scores and scores of the next player in some ordering of players. The planner has to choose one of the players based on the scores that they report for the next player additional independent reports that can be erroneous.
3. $n$ players. Each player's work has a score from distribution $F$. The player has a report of her own score that might be innacurate. The planner has to choose one of the players based on the scores that they report and additional independent reports that can be erroneous.
4. $n$ players. Each player's work has a score from distribution $F$. The player has a report of her own score and a score of the next player in some ordering of players that might be innacurate. The planner has to choose one of the players based on the scores that they report for the next player and additional independent reports that can be erroneous.

TODO: 
1. Implement all combinations of the above ideas in a robust way with the ability to test with different parameters and save outputs.
2. Train different models from above on the cluster.
3. Write down results as your master thesis.
4. Send draft to promotor.