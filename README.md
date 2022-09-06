# Assignment 3 - Sripad Joshi (joshisri), Karan Acharya (karachar), and Srimanth Agastyaraju (sragas)

*This is a report for Assignment 3 of the course Elements of Artificial Intelligence, CSCI-B 551, Fall 2021.*

*Created by Sripad Joshi (joshisri), MS in Computer Science, Karan Milind Acharya (karachar), MS in Computer Science and Srimanth Agastyaraju (sragas), MS in Data Science, Fall 2021.*

## Part 1: Part-of-speech tagging

### Approach

- Using the given training corpus, we calculate all the probabilities and other parameters which we'll be used to predict on a test set sentence.
- For each model, we predict using those pre-computed parameters and the given corresponding Bayes' Net architectures.
- For each word in the sentence, the tag will be predicted using each of the 3 models - Simple, HMM and Complex.

### Training the models

- We calculate all the possible parameters which we'll be required during inferences using all the 3 models.  
    Following is the approach:
    1. The same word can appear in the corpus as different POS tags. Hence, we calculate the number of times each unique word appears in the corpus classified by it's tag. (Stored in the dict: **vocabulary**). It'll look something like this:  
    ```{'word1': {'noun': 3, 'verb': 1}, 'word2':{'det': 12, 'x': 3}}```  
    2. Next, we calculate the transition counts and probabilities. This will store the counts of each tag followed by each tag and the corresponding probabilities respectively. These will be used by all the 3 models.
    3. Next, similarly the emission counts and probabilities. The counts and probabilities of each word in the training corpus given it's corresponding tag. These will also be used by all the 3 models.
    4. Similarly, we create 4 new dictionaries: **skip_emissions, skip_transitions, skip_emission_probs, and skip_transition_probs**. These will store the counts and probabilites of a each tag also being a child of it's grandparent, and each word being also a child of it's previous tag. These are required only for the *Complex* model.
    5. While calculating all the emissions - the normal and the skip, there are obvious instances where a word may not be associated with each of the 12 tags. e.g. The word *India* can only be a *noun*, and hence we can calculate the emission probability for this word being a noun being equal to the number of times the word appeared as a noun divided by the number of total noun instances. The emission probability of this word given other tags should be very low, hence we set an **alpha** value (0.01) as the count, and then calculate it's emission prob - which will be very low as compared to the *noun* tag. **In other words, we're penalising the probabilities for the words given a tag for which we know it has a very low chance of existing in reality**. (This is similar to the concept of [Laplace smoothing](https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece).)
    6. Next, we also calculate the prior probabilities of each of the 12 tags and store it in the dict **priors**.
- In summary, the training phase will only consist of creating and computing the dictionaries with probability parameters which would be used later while inferencing.

### Inference + Solution

### 1. Simple Bayes Net

- The idea for making predictions using the Simple Bayes Net in fig. 1(b) is as follows:
    1. There exist links only from a tag to it's corresponding word. No other links exist between successive tags and/or words.
    2. Thus, each word-tag pair is independent of every other pair. To predict the posterior probability of each tag given it's corresponding word, we use Bayes' Law.
    3. The output prediction for each word will be the argmax of the product of it's emission and prior probabilities. For words not in the training corpus, we simply use the priors and ignore the non-existent emission probability for that word. (We initially tried using a logic similar to the *Laplace idea* mentioned above, but it was giving poorer results, and hence we scrapped it.)

### 2. Using the Viterbi algorithm on HMM

- Here, as can be seen in fig. 1(a), we have richer dependencies between sucessive tags. It's a Hidden Markov Model and hence we can use the Viterbi algorithm which uses dynamic programming to find the most likely sequence of unobserved variables given observed variables.
- In the vanilla version of the algorithm, we fill the probability value directly in the Viterbi Table. As the sequence length increases, there's a possibility that the numbers will become 0 as the values become way way smaller. To avoid that pitfall, we use the logarithm of the values. (We still maximise, as we are using the positive log). All the products become sums.
- Initially only the 1st column is filled. And then every column until the last one is filled. With each filled value, we also store an integer value indicating the row which provided this value with the maximum. This will be later used when backtracking to find the path.
- For words not in the vocabulary, we don't consider the emission probabilities for those words and just the priors or the other terms are used. (Again here, we tried with using an estimate of an emission prob using the *Laplace* logic, but it was grossly underestimating the value for each tag. Ignoring the value completely led to better results.)
- Once the table is filled, we select the row in the last column with the max value and assign the tag for the last word corresponding to that row number. For all other columns, we simply use the integer value stored to trace back to the first word.

### 3. Complex model using MCMC-Gibbs Sampling

- Here, we perform Gibbs Sampling based on the Bayes' Net design given in fig. 1(c). This design involves transition links between alternate tags and emission links between a tag and it's next word as well. The sampling will be performed for each sentence in the test set.
- We referred to the lecture slides and the following additional webpages for in-depth understanding of the concept: [Link 1](http://vision.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf), [Link 2](https://www.cs.mcgill.ca/~dprecup/courses/ML/Lectures/ml-lecture08.pdf) and [Link 3](https://personal.utdallas.edu/~nrr150130/cs6347/2017sp/lects/Lecture_10_MCMC.pdf).
- The tags are the unobserved variables and the words are the observed variables. Following is the algorithm:  
    For each sentence,
    1. Initialise the 1st sample with random assignment of tags for each word in that sentence. (*We also tried by starting with a assignment of 'noun' for each tag. It had minimal difference in the final output.*)
    2. For n iterations,
        1. Assign the current sample to be the most recent one from the main list of samples.
        2. Pick an unobserved variable uniformly randomly. (*We can also run through all the tags(= length of sentence) in every iteration - this was how it was taught in the class. But this was taking a lot of computational time.*)
        3.  Sample Si from P(Si | S0, S1,...Si-1, Si+1, Si+2, ... Sn-1). We need to eliminate some variables. A variable is conditionally independent of all others given its Markov blanket (parents, children, spouses). Hence, when sampling we eliminate all others and just consider the probabilities associated with that variable. e.g. For S0, it doesn't have any parents. It's children are: W0, W1, S1, S2. Hence, we sample S0 based on only P(W0|S0), P(W1|S1), P(S1|S0), P(S2|S0) - These are the associated links for S0.
        4. Also depending on the variable being sampled, it may/may not have all parents and/or all children. These edge cases were considered. Please look at the code for better understanding. Again, for words not present in the training vocabulary, we didn't consider their non-existent emission probability.
        5. While sampling that variable, we consider all it's 12 possibilities corresponding to each tag. So, as if by flipping a 12-sided weighted biased coin, we assign a value to that variable while keeping the values of all other variables constant.
        6. This new sample is added to the main list of samples.
    3. The main list of n samples contains n tags for each word in the sentence. For every word, we just select the one that occurs the most and assign it as our prediction for that word.

## Conclusion

- The program completes under 10 minutes when run on the test set: *bc.test*. Following are the final results:

![Result](part1/p1-test-output.png)

- As evident from the results, the HMM model performs slightly better than the not-so-far-behind Complex and Simple models.

<hr/>

## **Part 2: Ice tracking**

**Aim:**
For a given radar echogram, detect the air-ice boundary and ice-bedrock boundary

**Assumptions:**
(as given in the a3-fa2021 pdf)
- **a**.The air-ice boundary is always above the ice-bedrock boundary by margin of say 10 pixels
- **b** Both these boundaries span across the image
- **c** The boundaries are assumed to be smooth along the columns

**Approaches for boundary detection**

### **simple (Bayes' Net)**

Here, we use the pre-implemented *edge_strength* method to get the edge strength. The stronger the edge(higher pixel value) the higher the probability of it being the boundary. So we use the edge strength matrix for a given image and convert the pixel values to probabilities (**emission probability**).

For a given image, for each column, we picked two row indices of highest probabilitiy values with a condition that they are atleast 10 pixels apart.  Of the two picked row indices, the value at lower row index represents the air-ice boundary pixel, the value at higher row index represent the ice-bedrock boundary pixel (because of the assumption **a**, which says air-ice boundary is always above ice-bedrock boundary)

Example results

| ![image](part2/results/09/air_ice_output_09_simple_ir.png) |
|:--:| 
| 09.png simple air-ice boundary |

| ![image](part2/results/09/ice_rock_output_09_simple_ai.png) |
|:--:| 
| 09.png simple ice-bedrock boundary |

But this approach fails for a difficult case where only using the edge strength is not enough for the boundary detection

| ![image](part2/results/23/air_ice_output_23_simple.png) |
|:--:| 
| 23.png simple air-ice boundary |

| ![image](part2/results/23/ice_rock_output_23_simple.png) |
|:--:| 
| 23.png simple ice-bedrock boundary |


### **HMM (Viterbi)**

Next we use viterbi algorith to solve for the boundaries.

For the viterbi algorithm - we use the same emission probabilities as before.

For transition probabilities, we take into account the assumption of *"smoothness"*

So, for a given row and column, the transition probability of going to the next column's same row is highest. The row above and below have lower transition probabilities. The transition probability for rows far below or above are set to 0.

* After calculating the air-ice boundary, used the part of the image below with the boundary to calculate the ice-bedrock boundary.

Example results

| ![image](part2/results/31/air_ice_output_31_viterbi.png) |
|:--:| 
| 31.png viterbi air-ice boundary |

| ![image](part2/results/31/ice_rock_output_31_viterbi.png) |
|:--:| 
| 31.png viterbi ice-bedrock boundary |

The difficult example which didn't work well with simple:

| ![image](part2/results/23/air_ice_output_23_viterbi.png) |
|:--:| 
| 23.png Viterbi air-ice boundary |

| ![image](part2/results/23/ice_rock_output_23_viterbi.png) |
|:--:| 
| 23.png Viterbi ice-bedrock boundary |

Here we see a huge improvement  over simple algorithm for image 23.png after using viterbi.


### **HMM (Viterbi) with human feedback**

In this approach, we use human feedback to further improve the boundary detection.

We divide the problem into two sub problems, one detecting the boundary from the human feedback point to the starting column backwards, other from the human feedback point to the last column in forward direction. Here, we set the initial probabilties for the viterbi algorithm such that the human feedback row has the highest initial probability value.
This improves the outputs further. Below are example outputs.


Example outputs:

| ![image](part2/results/30/air_ice_output_30_viterbi_human_feedback.png) |
|:--:| 
| 30.png viterbi - human feedback air-ice boundary |

| ![image](part2/results/30/ice_rock_output_30_viterbi_human_feedback.png) |
|:--:| 
| 30.png viterbi human feedback  ice-bedrock boundary |

The difficult example which didn't work well with simple:

| ![image](part2/results/23/air_ice_output_23_viterbi_human_feedback.png) |
|:--:| 
| 23.png Viterbi - human feedback air-ice boundary |

| ![image](part2/results/23/ice_rock_output_23_viterbi_human_feedback.png) |
|:--:| 
| 23.png Viterbi -  human feedback ice-bedrock boundary |

<hr/>

## **Part 3: Reading Text**

### **Task**

The task is to read the text data from an image, and outupt the data in the form of plain text. It is a typical example of an Optical Character Recognition problem. We are given a set of clean images of each character as a reference, and our task is to match a noisy image with the reference image as closely as possible, and return the closest possible character associated with the noisy character.

### **Approach**

### **Probability measures**

We are given the reference and noisy characters in the form of a mxn matrix, filled with '\*' and ' '. We can convert this matrix into a sparse matrix filled with 0 instead of ' ' and 1 instead of '\*'. I have flattened the matrix for a single pass of a list.

**1) Intersection over Union (IoU)**

We define the probability using IoU over the two flattened noisy and reference lists as follows:
(Defined as probability_2 in the code)

```python
probability_2 = (intersection count + 1) / (union count + 1).
```

**2) Dice Score**

I got this idea from google and wikipedia. Reference: <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>

My implementation of dice score is a bit modified:
(Defined as probability_3 in the code)

```python
probability_3 = (intersection count + 1) / (number_of_1s_in_reference + number_of_1s_in_noisy + 1)
```

In both the probabilistic measures, I have added + 1 to the numerator and denominator, in order to avoid log(0) or 0/0 errors.

### **Simple**

The noisy character (a_bar) and the reference character (a) are said to be similar if there is a maximum probability P such that P(noisy/reference) is maximum.

Hence for every character in the noisy sentence, we look through the entire reference set and then return the refernce character with the highest probability of P(noisy/reference).

Example: We are given a noisy a (a_bar). We do not know beforehand what it looks like. Hence we maximize the following: (From the problem statement).

![Simple Probability image from a3-fall2021.pdf](./part3/readme_images/simple_prob.png)

In our example, W can be defined as any reference character between [A-Za-z0-9(),.-!\'\"]

S in our example is the noisy character a_bar.
If the noisy character image is close to a known refernce character image, it outputs the plain text character associated with the reference pixels. Hence it outputs the answer as plain text character 'a', since 'a' is the most similar to a_bar.

However, this might not always be the case. For example, a noisy C (C_bar) could be more closer to 'G' than 'C'. Hence the model may output G as the answer, where a human might read it as a C given the image itself and the context of the letter.

The probability measure used for this problem is **Intersection over Union (IoU).**

### **Hidden Markov Model (HMM)**

We can define initial probabilities, transition probabilities and emission probabilities as follows:

1) Initial probabilities: The probability of a character occuring at the start.
    - Go through the train_file.txt supplied in the command line argument.
    - Count the frequency of each character and then divide it by the sum of frequencies to get the probability for each character.
    - Reduce the initial frequency of a space being the first character, since it is not likely to happen. In my case, I have divided its frequency by a thousand.
    - Divide the initial probability of each character by the sum of frequencies to obtain a probability measure.
    - Store the natural logarithm of this value and return the probability table.

2) Transition probabilities: The probability of the sentence having current character as c1 and next character as c2.
    - Create a nested dictionary where c1 is the key and the value is a dictionary. Let's call the value dictionary as temp_dict.
    - Now, the temp_dict contains c2 as the key and the value is the count of having c2 after c1 in the training set.
    - Divide the value each temp_dict with the sum of the values of the temp_dict to get the probability.
    - Go through the train_file.txt supplied in the command line argument.
    - Initialized the counts in each temp_dict as 0.0001 instead of 0, since it could cause a log error.
    - Store the natural logarithm of this value and return the probability table.

3) Emission probabilities: The probability that the characters of the train and test image are similar. It is a grid of probabilities where there is a probability for every character in the test images given the train images.
    - Create a nested dictionary where the state state_1 is the key and the value is a dictionary. Let's call the value dictionary as temp_dict.
    - Now, the temp_dict contains obs_1 as the key and the value is the probability that the state state_1 is similar to obs_1. The probability is calculated between state_1 and obs_1.
    - In this case, we are using the **Dice score**.
    - Store the natural logarithm of this value and multiply it by 50 (a random factor obtained through trial and error), so that the transition probabilities do not dominate the emission probabilities. Return this table.

HMM primarily balances two goals: maximizing the transition and emission probabilities. The viterbi algorithm code is taken from the solution provided for the in-class activity 2 on Oct 20, 2021. There is a slight modification though. Since we are calculating the natural logarithm for each of the probabilities, we maximize over the sum of transition and emission probabilities instead of the product.

## Results

We have typed out the actual answer by reading from the test images (human input). These strings are the expected answers.
Accuracy is measured as follows:
Accuracy = (Number of characters matched between the output and the expected answer) / (Total number of characters in the expected answer).

The below output can be achieved using the following command:

```bash
python3 my_own_test_script.py ./test_images/courier-train.png ../part1/bc.train ./test_images/test-0-0.png
```

Overall accuracy for simple model is **89.39%**
Overall accuracy for Hidden Markov Model is **94.69%**

![result 1](./part3/readme_images/result_1.png)
![result 2](./part3/readme_images/result_2.png)
![result 3](./part3/readme_images/result_3.png)
![result 4](./part3/readme_images/result_4.png)
![result 5](./part3/readme_images/result_5.png)
![result 6](./part3/readme_images/result_6.png)
![result 7](./part3/readme_images/result_7.png)

