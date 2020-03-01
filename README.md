# Bartend-A.I.
<p align="center">
  <img src=Data/images/robot.0.jpg>
</p>

> By: Haydin Bradshaw



## Motivation & Goal:

Applications of deep learning and neural networks have impacted and improved practically every industry they touch. From manufacturing, information technology, business, and engineering, the application of large data and innovative algorathims have lead to astounding improvements in these fields. Another field that could utilize the benefits of deep learning and specifically in the vein of "creativity" is the food and beverage industry.

The goal of this project is to implement the advances in recurrent neural networks to mixology/bartending. For anyone who enjoys indulging in or creating cocktails, they realize the breadth of creativity and inspiration required of a bartender to devise new drinks. However, even the best bartender's creative well will eventually run dry. When that happens, neural networks, specifically attention augmented long short-term recurrent neural networks can be used to help. These neural networks can be implemented to learn the "style" of a bartender's recipes and "creatively" generate new recipes to reinspire him.

## Data:

#### The Dataset:

For neural networks a large amount of data is a must. To fulfill my goal of building a neural network a large dataset of cocktail recipes was necessary. Credit to [*odditysoftware*](https://http://www.odditysoftware.com/page-datasales7.htm/) for the drinks and cocktails dataset (I have no right to distribute said dataset so only descriptions and small examples will be provided). The dataset is comprised of 16,351 rows and 8 columns. The columns are id, d_name, d_cat (category), d_alcohol, d_glass, d_ingredients, d_instructions, d_shopping. 

Here's an example of the data:
![](/Data/images/data_example.png)

#### Preprocessing & EDA:

The primary data needed to generate new recipes are the ingredients of the cocktails. 
