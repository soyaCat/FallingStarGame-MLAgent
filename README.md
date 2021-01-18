# FallingStarGame-MLAgent

# 1.Purpose of this repositories
Before using ML_agent-Release12 and pytorch, It is important for me to make easy to use them

I have used ML_Agent 0.8.1 and Tensorflow in the past, and familiar with the style of writing code using them,  
I created this repository with goal that allowing me to write code in similar code style when using the latest versions of ML_Agent 

With this purposes, I Implement 
1. multi Agent Environment
2. machine learning model that using stacked vector observation information
3. machine learnig model that using stacked visual observation information

and try to 
1. seperate function that are likely to be recycled later
2. By using black box code writing method, allow me to write code without worring about data format conversion
3. The number of variables is automatically changed accoding to number of agents

Because this code train model using vector observation and model using visual observaion at same time,  
the program is heavy compared to trained targets.  
And it uses a large amount of memory because it accumulate stacked visual observation in memory

# Environment
This is my environment
  - Unity 2020.1.3f1
  - MLAgents Release 12
  - python 3.8.5
  - MLAgents API ver 0.23.0
  - pytorch 1.7.0
  
 # How to use it?
 
