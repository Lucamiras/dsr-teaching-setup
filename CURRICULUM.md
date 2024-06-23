# The DSR Roadmap

This document contains a detailed overview over the curriculum and roadmap for the current batch of Data Science Retreat Berlin.

1. Technical Fundamentals
2. Data Science & Machine Learning Fundamentals
3. Mini Competition
4. Deep Learning / Generative AI
5. Practical Data Science
6. Soft skills
7. The Portfolio Project

## Technical Fundamentals

### Numpy & Pandas (Online - voluntary prep)
#### Purpose
Essential for efficiently managing, analysing, and manipulating data. The foundational elements necessary for advanced data science applications.

#### Python
- Data types: Numbers, Strings, Lists, Dictionaries, Tuples.
- Control with if/for/while, iterations, functions, and lambdas.
- File operations and data type nuances.

#### Numpy
- Array essentials: Creation, attributes, and operations.
- Key functions: reshaping, Ufuncs (np.add), broadcasting, advanced indexing.

#### A brief introduction to Pandas
- Data structures: Series and DataFrames
- Data handling: indexing, missing data, Ufuncs
- Data manipulation: combining, grouping, pivoting

### Git & Bash
#### Purpose
Reproducibility and organization of code

#### Bash
- What: Bash is a shell for accessing and controlling various tools and services. Bash allows you to navigate your file system, install and manage packages, run scripts, and interact with other tools like Anaconda, Python, and Git.
- Why: It provides fast access to important tools for Data Science such as Anaconda and Python
- Focus: Basic Bash command line usage and file system navigation.
- Activities: Hands-on exercises including file operations, text manipulation, and how the shell interacts with Python scripts.

#### Git
- What: Git is a version control tool
- Why: It allows for easy collaboration with other developers
- Focus: Introduction to Git fundamentals and version control concepts.
- Activities: Theory mixed with practical exercises on basic Git commands and collaborative features.

*Note: Adjustments to content pacing based on class response to ensure comprehension.*

### Docker
Get to know docker as tool enabling consistent and scalable environments for data science projects, simplifying collaboration and deployment.
- Learn to navigate and utilize Docker for deploying containerized applications. 
- Work with existing Docker images to understand container functionality.
- Build custom Docker images, integrating data science tools and environments.

### SQL
Manage relational databases effectively
- Syntax Mastery: Practice with SQLBolt, HackerRank, and quizzes.
- Advanced Application: Explore Google BigQuery 

### Probability & Statistics
Grasp probability’s role in statistical reasoning. Employ statistics for insightful data analysis.

#### Probability Essentials (Online - voluntary prep)
- Core Concepts: Probability, Random Variables, Distributions.
- Calculations: Marginal, Conditional Probability, Chain Rule.
- Relationships: Independence, Expectation, Variance, Covariance.
- Advanced: Common Distributions, Bayes' Rule, Information Theory, Monte Carlo, Markov Chains.

#### Statistics Fundamentals
- Basics: Descriptive Statistics, Combinatorics.
- Deep Dive: Distributions, Sampling, Hypothesis Testing, Model Estimation.

### Visualization
Learn to craft web pages, render interactive charts, and build web apps efficiently.

- D3 Module: Master interactive visualizations using JavaScript, handling the DOM (Document Object Model), SVG (Scalable Vector Graphics) /CANVAS (HTML Canvas element for bitmap rendering), and data dynamics.
- Plotly Module: Create advanced visualizations with Plotly Express and develop web apps with Plotly DASH.

## DS & ML Fundamentals

### Machine Learning Fundamentals
Foundational tools to construct, evaluate, and refine ML models for real-world application

#### Machine Learning Overview
- What ML is (not) able to achieve
- The ML Project lifecycle
- Machine learning post-ChatGPT

#### Data Understanding
- Labeled vs unlabeled data (Supervised vs Unsupervised Learning)
- Structured vs unstructured data (Tables vs Images/Text/Sound)

#### Core ML Concepts and Techniques
- Loss functions
- Distance Metrics
- Embeddings aka representation learning
- Performance metrics vs loss functions
- The bias vs variance tradeoff
- Evaluating performance with train, test, and validation sets
- Hyperparameters vs parameters
- Gradient descent

#### Supervised Learning
- Tasks of supervised learning (regression and classification)
- Baseline models
- Algorithms for Supervised learning:
- Linear and logistic regression
- Decision trees, random forests, and gradient boosting
- Neural networks

#### Unsupervised Learning
- Tasks of unsupervised learning (clustering and dimensionality reduction)
- Algorithms for Unsupervised learning:
- K-means
- Principal component analysis

### Object-oriented programming & Software architecture
To ensure the development of maintainable, scalable, and reliable data science systems with high code quality and efficiency.

#### OOP Fundamentals
- Utilize classes and objects for modelling data science problems.
- Implement robust logging to track the execution and issues.
- Develop error handling strategies to manage exceptions gracefully.

#### Testing Approaches
- Write acceptance tests to validate software meets business requirements.
- Conduct unit tests to ensure individual components function correctly.

#### Software Architecture Concepts
- Adopt software architecture patterns that suit data science and AI projects, facilitating future growth and integration.

### DS Fundamentals
Develop a structured idea of the data science workflow. 

#### Visualization Techniques
- Mastering Matplotlib for data representation.
- Dimensionality reduction with t-SNE and PCA for insightful visuals.

#### Data-Cleaning Essentials
- In-place column cleaning methodologies.
- Necessary steps for dataset preparation.

#### Feature Engineering Strategies
- Techniques for crafting new, informative features.
- Enhancements to boost model performance.

#### Linear Models and Beyond
Introduction to linear modelling techniques.

#### Model Selection & Evaluation
- Guidelines for experimenting with different models.
- Criteria for model selection and hyperparameter optimization.

#### Interpretation and Feature Selection
- Understanding linear model coefficients and applying LIME for interpretability.
- Univariate and stability selection methods for feature importance.

### Trees
Foundational tree models for classification and regression tasks.

#### Ensemble Methods - Strength in Numbers:
- Bagging: Combining multiple trees to reduce variance and improve stability.
- Random Forest: A bagging technique with a forest of decision trees for robust predictions.
- Boosting: Sequentially building trees to correct previous errors and enhance performance.
- Adaboost & Gradient Boosted Trees: Specialized boosting methods focusing on different aspects of error reduction.

#### Advanced Techniques
- Encoding Categorical Data: Techniques like mean, label, and target encoding to transform categorical variables for improved tree and ensemble model performance.

### Time series (Online)
To enable participants to understand, model, and forecast data that change over time, providing insights that are crucial for decision-making in various domains.

#### Understanding Time Series Data
- Fundamentals: Tasks in Time Series Analysis and unique challenges.
- Key Properties: Explore trend, seasonality, and noise within time series data.

#### Analytical Techniques
- Autocorrelation Studies: Delve into Autocorrelation and Partial Autocorrelation to understand data dependencies.
- Decomposition: Break down time series into its components to analyse separately.

#### Smoothing Methods
- Moving Averages and Exponential Smoothing: Techniques to smooth out short-term fluctuations and highlight longer-term trends or cycles.

#### Predictive Techniques
- Baseline Approaches: Next day prediction, Moving Averages, Exponential Moving Averages.
- Classical Models: ARIMA, Holt-Winters method.
- Modern Approaches: Leveraging Machine Learning for advanced time series forecasting.

#### Toolkits for Implementation
- SkTime & Darts: Introduction to specialized libraries for time series analysis and modelling.

## Mini Competition
Work effectively in a team – both technically (Git & Bash) & conceptually (task distribution, tackling dependencies, data strategy).

The difference stages of a data science project – data exploration, data cleaning & preparation, feature engineering, model choice, model training, model testing & results delivery.

Working on a realistic project is great practice for the real world.

## Deep Learning / Generative AI

### Backpropagation
- Master the backpropagation algorithm from the basics of perceptrons to advanced multilayer networks.
- Implement Forward (calculate outputs using activation functions) and Backward Pass (calculate gradients using the chain rule)
- Implement neural network training and optimization using Python.

### Practical Reasoning for AI
- Sharpen skills in argument analysis and applied logic to understand and construct strong logical connections within AI systems
- Use cases: prompting, construction of RAG solutions / Agents

### Complexity Theory in Machine Learning
#### Computational complexity
- Big O Notation (upper bound of an algorithm's running time or space requirements in terms of the size of the input data.)
- Trade-offs between memory and computation in algorithms & data structures 
- Learn to select & optimise AI models for efficiency, sclability, and real-world deployment

### NLP, Transfer Learning & Representation (TensorFlow) (everything from scratch)
- Deep dive in (word) embeddings - Representations and what they encode, how they're created, and why they perform so well!
- Attention-based models - Attention is all you need
- Build your own GPT and generate text.

### Image Processing (TensorFlow)
- Basic Image classification using convolutional neural networks (CNNs)
- Transfer Learning with CNNs
- Region-based CNNs for object detection tasks
- Region of Interest Identification and Intersection over Union (IoU) as a metric for evaluating object detection 
models
- U-Net architecture for image segmentation tasks
- GoogLeNet (Inception architecture) for image classification 

### Computer Vision (PyTorch)
- Image similarity
- Image data pipelining - processing, augmentation, shuffling, batching
- Training classifiers with FastAI
- Object detection & image segmentation using Meta’s Detectron2

### Deep Reinforcement Learning
- Concentrates on teaching agents to make decisions by trial and error. Typically applied in areas like robotics, autonomous systems, game playing, and real-time decision-making in finance
- Core Principles of RL - understand RL's fundamentals, including agents, environments, actions, states, rewards, and policies, and how they guide decision-making through trial and error.
- Practical Frameworks - explore RL's implementation in various sectors using frameworks like OpenAI Gym, focusing on applications in robotics, gaming, and finance.
- Advanced Techniques and Trends - delve into the mechanisms and recent developments of RL, covering both theoretical methods like value and policy iteration and cutting-edge advancements like deep RL and multi-agent systems.

### Geometric Deep Learning
- Focus on GDL’s application in Graph Neural Networks (GNNs). Commonly applied in social network analysis, 3D shape processing, and molecule structure prediction.  
- Architecture & functioning of various GNNs including Graph Convolutional Networks (GCNs)
- Attention-based GNNs
- Message Passing Neural Networks (MPNNs)
- Equivariant GNNs, understanding how they maintain symmetry and consistency in data representations.

### Debugging Deep Learning Models
- Most common challenges & bugs with Deep Learning Models
- Intuition on how to identify & treat arising issues with tunning code in deep learning
- LLM-base debugging approach will be shown in detail 
- A small hand-on project to apply the newly acquired skills

### LLM Fine-tuning in Google Colab (online)
#### Typical Architecture & Steps 
- Model Quantization: Quantization reduces a model's precision by converting floating-point numbers to lower-bit representations, enhancing computational efficiency and reducing memory footprint.
Example: Deploying a quantized neural network on a mobile device to enable real-time image recognition with limited hardware resources.
- Flash Attention: Flash Attention is a technique to optimize the attention calculation in transformers for improved speed and efficiency in processing sequences.
Example: Using Flash Attention to speed up language translation tasks in an AI-powered chatbot, resulting in quicker response times.
- RoPE (Rotary Positional Embeddings): RoPE integrates positional information into attention mechanisms, allowing the model to better capture the order and relationship between elements in a sequence.
Example: Implementing RoPE in a text generation model to more accurately reflect syntactic structures based on word positions.
- Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA): LoRA is a method for adapting large pre-trained models to specific tasks without extensively altering the original parameters, focusing on efficiency and adaptability.
Example: Fine-tuning a pre-trained language model for a legal document analysis task using LoRA to introduce task-specific adjustments while retaining the model's extensive general knowledge.

### Retrieval Augmented Generation (RAG)
A technique combining retrieval of information with generative models for advanced NLP tasks.
- RAG Components and Functionality: Explore RAG's system design including text embedding, vector storage, retrieval, and optimization strategies.
- Practical Application: Engage with various tools and frameworks for RAG, addressing bias mitigation and ethical considerations in NLP.
- If you’d like, check https://www.langchain.com/ & https://www.llamaindex.ai/ 

## Practical Data Science

### Building apps with Streamlit
- Introduction to Streamlit: Gain a foundational understanding of Streamlit for web application deployment in data science.
- Project Deployment: Learn to transform data science projects into interactive web applications.
- Hands-On Experience: Apply your knowledge by building and deploying a small project with Streamlit by the end of the session.

### ML Ops
- FastAPI and VertexAI: Learn to deploy machine learning models using FastAPI and manage ML pipelines with VertexAI.
- Preparation: Ensure a Google Cloud Platform account is set up prior to the session for practical deployment exercises.

### Test-driven development
- TDD Fundamentals: Master the TDD cycle (Red-Green-Refactor), understand its benefits in software development, and learn the distinctions between TDD and Behavior-Driven Development (BDD).
- Testing Techniques: Gain proficiency in writing, organizing, and managing test cases including unit and integration testing, and applying testing best practices.
- Practical TDD Applications: Engage in hands-on exercises to implement TDD on Python projects, including writing tests that initially fail and refactoring code to pass tests.
- Advanced TDD with GitLab: Utilize GitLab for issue tracking, milestone management, and setting up continuous integration and deployment pipelines.
- Best Coding and Testing Practices: Explore best practices in coding and test-driven development, including the use of code formatters and the Python Enhancement Protocol (PEP).

## Soft Skills

### Business Communication
- Learn to tell a compelling data story
- Communication with stakeholder
- Job interview prepration

### Career Support
- Career advice
- CV support
- Job interview preparation

## The Portfolio Project
Data science is about doing and showcasing your skills with meaningful projects: 
- You found a topic, for which data science offers a solution / optimisation / invention
- You dealt with the issue of finding or creating data
- You delivered a satisfying result

#### Project formation process:
- Seek communication as soon as possible
- Brain-storm with each other
- The idea should be yours - you are expected to be creative 
- Provide an abstract (1 page / 1 paragraph) to Arun. Several advantages: 
    - You will have more out of your project and will save time and nerves ;)  if your project idea is similar to a past project, Arun will put you in touch with the respective team
    - Gives you exposure to potential employers   there is the possibility to reach out to companies, if you’d like to do it in cooperation with one.
    - Once you have decided to do the project this way, you are committed!  
    - Better mentor match  Arun has enough time to reach out to mentors 

Mentors  - common mentoring (usually Mondays & Thursdays) and separate mentors, if a project topic was picked on time