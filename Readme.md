\# Q-Learning on Gymnasium FrozenLake



A Python implementation of tabular \*\*Q-Learning\*\* using OpenAI Gymnasium's `FrozenLake-v1` environment. It explores training autonomous agents in both \*\*deterministic\*\* and \*\*stochastic (slippery)\*\* grid environments using custom reward engineering and epsilon-greedy decay\[cite: 2, 3].



\---



\## 📌 Features



\- \*\*Custom Reward Design:\*\* Optimized rewards for goal achievement, step penalties, and hazard avoidance\[cite: 2, 3].

\- \*\*Deterministic vs. Slippery Modes:\*\* Compares policy performance on fixed paths vs. probabilistic ice slipping\[cite: 2, 3].

\- \*\*Visual Demo:\*\* Renders live episode evaluation using Pygame (`render\_mode='human'`)\[cite: 2, 3].



\---



\## 📂 Repository Structure



.

├── frozenlake\_slippery.py        # Q-Learning on slippery environment (is\_slippery=True)

├── frozenlake\_deterministic.py   # Q-Learning on deterministic environment (is\_slippery=False)

├── requirements.txt              # Project dependencies\[cite: 2, 3]

└── README.md                     # Project documentation  

\---



\## 🚀 Quick Start



1\. \*\*Install Dependencies:\*\*

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

Run Deterministic Agent:Bashpython frozenlake\_deterministic.py

Run Slippery Agent:Bashpython frozenlake\_slippery.py

👨‍💻 AuthorAhmed Sherkawy - Computer Science \& AI Student

