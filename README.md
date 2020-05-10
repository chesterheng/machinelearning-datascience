# The Complete Junior to Senior Web Developer Roadmap (2020)

## Table of Contents

- [The Complete Junior to Senior Web Developer Roadmap (2020)](#the-complete-junior-to-senior-web-developer-roadmap-2020)
  - [Table of Contents](#table-of-contents)
  - [**Section 2: Machine Learning 101**](#section-2-machine-learning-101)
  - [**Section 3: Machine Learning and Data Science Framework**](#section-3-machine-learning-and-data-science-framework)
  - [**Section 4: The 2 Paths**](#section-4-the-2-paths)
  - [**Section 5: Data Science Environment Setup**](#section-5-data-science-environment-setup)
  - [**Section 6: Pandas: Data Analysis**](#section-6-pandas-data-analysis)
  - [**Section 7: NumPy**](#section-7-numpy)
  - [**Section 8: Matplotlib: Plotting and Data Visualization**](#section-8-matplotlib-plotting-and-data-visualization)
  - [**Section 9: Scikit-learn: Creating Machine Learning Models**](#section-9-scikit-learn-creating-machine-learning-models)
  - [**Section 10: Supervised Learning: Classification + Regression**](#section-10-supervised-learning-classification--regression)
  - [**Section 11: Milestone Project 1: Supervised Learning (Classification)**](#section-11-milestone-project-1-supervised-learning-classification)
  - [**Section 12: Milestone Project 2: Supervised Learning (Time Series Data)**](#section-12-milestone-project-2-supervised-learning-time-series-data)
  - [**Section 13: Data Engineering**](#section-13-data-engineering)
  - [**Section 14: Neural Networks: Deep Learning, Transfer Learning and TensorFlow 2**](#section-14-neural-networks-deep-learning-transfer-learning-and-tensorflow-2)
  - [**Section 15: Storytelling + Communication: How To Present Your Work**](#section-15-storytelling--communication-how-to-present-your-work)
  - [**Section 16: Career Advice + Extra Bits**](#section-16-career-advice--extra-bits)
  - [**Section 17: Learn Python**](#section-17-learn-python)
  - [**Section 18: Learn Python Part 2**](#section-18-learn-python-part-2)
  - [**Section 19: Bonus: Learn Advanced Statistics and Mathematics for FREE!**](#section-19-bonus-learn-advanced-statistics-and-mathematics-for-free)
  - [**Section 20: Where To Go From Here?**](#section-20-where-to-go-from-here)
  - [**Section 21: Extras**](#section-21-extras)

## **Section 2: Machine Learning 101**

- What Is Machine Learning?
  - Machines can perform tasks really fast
  - We give them instructions to do tasks and they do it for us
  - Computers used to mean people who do tasks that compute
  - Problem: How to get to Danielle's house using Google maps?
  - Imagine we had ten different routes to Danielle's house
    - Option 1: I measure each route one by one 
    - Option 2: I program and tell the computer to calculate these 10 routes and find the shortest one.
  - Problem: Somebody left a review on Amazon. Is this person angry?
  - How can I describe to a computer what angry means?
  - We let machines take care of the easier part of which things we can describe
  - Things that are hard to just give instructions to, we let human do it
  - The goal of machine learning is to make machines act more and more like humans because the smarter they
- [AI/Machine Learning/Data Science](A Beginner’s Guide To Data Science)
  - AI: machine that acts like human
  - Narrow AI: machine that acts like human at a specific task
  - General AI: machine that acts like human with multiple abilities
  - Machine Learning: a subset of AI
  - Machine Learning: an approach to achieve artificial intelligence through systems that can find patterns in a set of data
  - Machine Learning: the science of getting computers to act without being explicitly programmed
  - Deep Learning: a subset of Machine Learning
  - Deep Learning: one of the techniques for implementing machine learning
  - Data Science: analyzing data and then doing something with a business goal
- [Teachable Machine](https://teachablemachine.withgoogle.com/)
- How Did We Get Here?
  - Goal: Make business decisions
  - Spreadsheets -> Relational DB -> Big Data (NoSQL) -> Machine Learning
    - Massive amounts of data
    - Massive improvements in computation
  - Steps in a full machine learning project
    - Data collection (hardest part) -> Data modelling -> Deployment
    - Data collection
      - How to clean noisy data?
      - What can we grab data from?
      - How do we find data?
      - How do we clean it so we can actually learn from it?
      - How to turn data from useless to useful?
    - Data modelling
      - Problem definition: What problem are we trying to solve?
      - Data: What data do we have?
      - Evaluation: What defines success?
      - Features: What features should we model?
      - Modelling: What kind of model should we use?
      - Experiments: What have we tried / What else can we try?
- [Machine Learning Playground](https://ml-playground.com)
- [Types of Machine Learning](http://vas3k.com/blog/machine_learning/): predict results based on incoming data
  - Supervised: Data are labeled into categories
    - classification: is this an apple or is this a pear?
    - regression: based on input to predict stock prices
  - Unsupervised: Data don't have labels
    - clustering: machine to create these groups
    - association rule learning: associate different things to predict what a customer might buy in the future
  - Reinforcement: teach machines through trial and error 
  - Reinforcement: teach machines through rewards and punishment
    - skill acquisition
    - real time learning
- What Is Machine Learning? Round 2
  - Now: Data -> machine learning algorithm -> pattern
  - Future: New data -> Same algorithm (model) -> More patterns
  - Normal algorithm: Starts with inputs and steps -> Makes output
  - Machine learning algorithm
    - Starts with inputs and output -> Figures out the steps
  - Data analysis is looking at a set of data and gain an understanding of it by comparing different examples, different features and making visualizations like graphs
  - Data science is running experiments on a set of data with the hopes of finding actionable insights within it
    - One of these experiments is to build a machine learning model
  - Data Science = Data analysis + Machine learning
- Section Review
  - Machine Learning lets computers make decisions about data
  - Machine Learning lets computers learn from data and they make predictions and decisions
  - Machine can learn from big data to predict future trends and make business decision

**[⬆ back to top](#table-of-contents)**

## **Section 3: Machine Learning and Data Science Framework**
- Introducing Our Framework
  - Focus on 
    - Practical solutions
    - Writing machine learning code
  - Create a framework
  - Match to data science and machine learning tools
  - Learn by doing
- [6 Step Machine Learning Framework](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/)
  - Problem definition: What problems are we trying to solve?
    - Supervised or Unsupervised
    - Classification or Regression
  - Data: What kind of data do we have?
    - Structured or Unstructured
  - Evaluation: What defines success for us?
    - Example: House data -> Machine learning model -> House price 
    - Predicted price vs Actual price
  - Features: What do we already know about the data?
    - Example: Heart disease? Feature: body weight
    - Turn features such as weight into patterns to make predictions whether a patient has heart disease?
  - Modelling: Based on our problem and data, what model should we use?
    - Problem 1 -> Model 1
    - Problem 2 -> Model 2
  - Experimentation: How could we improve/what can we try next?
- Types of Machine Learning Problems
  - When shouldn't you use machine learning?
    - Will a simple hand-coded instruction based system work?
  - Main types of machine learning
    - Supervised Learning: data and label -> make prediction
      - Classification: Is this example one thing or another?
        - Binary classification = two options 
        - Example: heart disease or no heart disease?
        - Multi-class classification = more than two options
      - Regression: Predict a number
        - Example: How much will this house sell for?
        - Example: How many people will buy this app?
    - Unsupervised Learning: has data but no labels
      - Existing Data: Purchase history of all customers
      - Scenario: Marketing team want to send out promotion for next summer
      - Question: Do you know who is interested in summer clothes?
      - Process: Apply labels such as Summer or Winter to data
      - Solution: Cluster 1 (Summer) and Cluster 2 (Winter)
    - Transfer Learning: leverages what one machine learning model has learned in another machine learning model
      - Example: Predict what dog breed appears in a photo
      - Solution: Find an existing model which is learned to decipher different car types and fine tune it for your task
    - Reinforcement Learning: a computer program perform some actions within a defined space and rewarding it for doing it well or punishing it for doing poorly
      - Example: teach a machine learning algorithm to play chess
  - Matching your problem
    - Supervised Learning: I know my inputs and outputs
    - Unsupervised Learning: I am not sure of the outputs but I have inputs
    - Transfer Learning: I think my problem may be similar to something else

**[⬆ back to top](#table-of-contents)**

## **Section 4: The 2 Paths**

**[⬆ back to top](#table-of-contents)**

## **Section 5: Data Science Environment Setup**

**[⬆ back to top](#table-of-contents)**

## **Section 6: Pandas: Data Analysis**

**[⬆ back to top](#table-of-contents)**

## **Section 7: NumPy**

**[⬆ back to top](#table-of-contents)**

## **Section 8: Matplotlib: Plotting and Data Visualization**

**[⬆ back to top](#table-of-contents)**

## **Section 9: Scikit-learn: Creating Machine Learning Models**

**[⬆ back to top](#table-of-contents)**

## **Section 10: Supervised Learning: Classification + Regression**

**[⬆ back to top](#table-of-contents)**

## **Section 11: Milestone Project 1: Supervised Learning (Classification)**

**[⬆ back to top](#table-of-contents)**

## **Section 12: Milestone Project 2: Supervised Learning (Time Series Data)**

**[⬆ back to top](#table-of-contents)**

## **Section 13: Data Engineering**

**[⬆ back to top](#table-of-contents)**

## **Section 14: Neural Networks: Deep Learning, Transfer Learning and TensorFlow 2**

**[⬆ back to top](#table-of-contents)**

## **Section 15: Storytelling + Communication: How To Present Your Work**

**[⬆ back to top](#table-of-contents)**

## **Section 16: Career Advice + Extra Bits**

**[⬆ back to top](#table-of-contents)**

## **Section 17: Learn Python**

**[⬆ back to top](#table-of-contents)**

## **Section 18: Learn Python Part 2**

**[⬆ back to top](#table-of-contents)**

## **Section 19: Bonus: Learn Advanced Statistics and Mathematics for FREE!**

**[⬆ back to top](#table-of-contents)**

## **Section 20: Where To Go From Here?**

**[⬆ back to top](#table-of-contents)**

## **Section 21: Extras**

**[⬆ back to top](#table-of-contents)**
