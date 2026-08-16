Entropy cluster.
https://entropy-doc.mimuw.edu.pl/registeringusers.html

Articles:
https://arxiv.org/pdf/1706.03459
Selecting a winner with external referees

https://www.ifaamas.org/Proceedings/aamas2018/pdfs/p354.pdf :
RegretNet with bic constraints. There is empirical "interim" regret constrained

https://www.ijcai.org/proceedings/2018/0036.pdf
https://www.ijcai.org/Proceedings/16/Papers/068.pdf

https://arxiv.org/abs/2511.11157

Notebook:
https://colab.research.google.com/drive/1XNDCkFNG3KNA9l9b1AuUKcAfkDmmmbp_?usp=sharing

Project text:
https://www.overleaf.com/project/68fe728d5787c7dcef1d77bc

This repository has code for representing games similar to the game from “Selecting a winner with external referees”,
and mechanisms for making decisions as torch models. These mechanisms can be trained, which is the main purpose
the repository.

The training can be started with the following commands:

Basic testing command:
```bash
python main.py --train_dataset_size 16 --eval_dataset_size 16 --players_num 3 --epoch_num 10 --device cpu --model_path checkpoints/basic_test.pt
```

Default command:
```bash
python main.py 
```

Command for Two players, inverted error, self report:
```bash
python main.py --players_num 2 --player_report_class invert_value --strategy SELF_REPORT 
```

Command for Five players, inverted error, self report:
```bash
python main.py --players_num 5 --player_report_class invert_value --strategy SELF_REPORT 
```

Command for Two players, inverted error, self report:
```bash
python main.py --players_num 2 --player_report_class invert_value --strategy OTHER_REPORT 
```

Command for Five players, inverted error, self report:
```bash
python main.py --players_num 5 --player_report_class invert_value --strategy OTHER_REPORT 
```

## Specifics of training

## Metrics recorded
1. Misreport gradient ascent plots: (where misreports are, heatmap of score, where true report is) - can be done at inference
2. Heatmap of allocation for two players - can be done at inference
3. Bar graph of regret and revenue for different architectures on the same problem - can be done at inference
4. Compare with optimal design when it is known - can be done at inference
5. rgt(90%) = value of regret that such that for 90% of valuations lower is achieved - can be done at inference


Ideas:
Ideas for score misreporting:
1. Error of reporting means that players score gets reported as $1 - s$, where $s$ is the actual score. The score gets misreported with some set probability smaller than $\frac{1}{2}$.
2. Error of reporting means that player’s score gets reported as $s + \frac{1}{2} \mod 1 $, where $s$ is the actual score. The score gets misreported with some set probability smaller than $\frac{1}{2}$.
3. Reported scores are actual scores with random error term from set distribution added.
4. Reported scores are actual scores with random error term from set distribution added and then normalised back into correct range.

Situations:
1. $n$ players. Each player’s work has a score from distribution $F$, that is known to the player. The planner has to choose one of the players based on the scores that they report and additional independent reports that can be erroneous.
2. $n$ players. Each player’s work has a score from distribution $F$. Players know their own scores and scores of the next player in some ordering of players. The planner has to choose one of the players based on the scores that they report for the next player additional independent reports that can be erroneous.
3. $n$ players. Each player’s work has a score from distribution $F$. The player has a report of her own score that might be innacurate. The planner has to choose one of the players based on the scores that they report and additional independent reports that can be erroneous.
4. $n$ players. Each player’s work has a score from distribution $F$. The player has a report of her own score and a score of the next player in some ordering of players that might be innacurate. The planner has to choose one of the players based on the scores that they report for the next player and additional independent reports that can be erroneous.

TODO: 
1. Implement all combinations of the above ideas with the ability to test with different parameters and save outputs.
2. Train different models from above on the cluster.
3. Write results as your master thesis.
4. Send draft to promotor.

# TODO NOW:
- List of commands for training

# TOOD GENERAL
- Check the specs, for how many training epoch we want
- Slurm script for training on the cluster
- Testing small cases locally
- Testing on entropy cluster
- Run on entropy cluster
- Prepare the document with explanation of approach 
- -Add evaluation functions:
  -  Misreport gradient ascent plots: (where misreports are, heatmap of score, where true report is) - can be done at inference 
  - Heatmap of allocation for two players - can be done at inference
  - Bar graph of regret and revenue for different architectures on the same problem - can be done at inference
  - Compare with optimal design when it is known - can be done at inference
  - rgt(90%) = value of regret that such that for 90% of valuations lower is achieved - can be done at inference
- Add results graphs to document
- Implement the theoretical best mechanism from "Selecting a winner with external referees"
- Add a beta (bimodal distribution) to draw values from.