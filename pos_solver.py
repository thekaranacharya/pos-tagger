###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
# Karan Milind Acharya [karachar]
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    tag_counts, vocabulary, priors = {}, {}, {}
    transitions, transition_probs, emission_probs = {}, {}, {}
    skip_transitions, skip_transition_probs, skip_emissions, skip_emission_probs = {}, {}, {}, {}
    alpha = 0.01
    simple_output = None

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, labels):
        if model == "Simple":
            result = 0

            # All priors
            for label in labels:
                # print(math.log(self.priors[label]))
                result += math.log(self.priors[label])

            for word, label in zip(sentence, labels):
                # if word in vocab and label present for that word
                if word in self.vocabulary.keys():
                    # print(math.log(self.emission_probs[word][label]))
                    result += math.log(self.emission_probs[word][label])

            # return -999
            return result
        elif model == "HMM":
            result = 0

            # Just 1 prior
            result += math.log(self.priors[labels[0]])

            # All transition probabilities
            for i in range(1, len(labels)):
                result += math.log(self.transition_probs[labels[i]][labels[i - 1]])

            # All emission probabilities
            for word, label in zip(sentence, labels):
                if word in self.vocabulary.keys():
                    result += math.log(self.emission_probs[word][label])

            return result
        elif model == "Complex":
            result = 0

            # Just 1 prior
            result += math.log(self.priors[labels[0]])

            # All transition probabilities
            for i in range(1, len(labels)):
                result += math.log(self.transition_probs[labels[i]][labels[i - 1]])

            # All emission probabilities
            for word, label in zip(sentence, labels):
                if word in self.vocabulary.keys():
                    result += math.log(self.emission_probs[word][label])

            # All skip-transition probabilities
            for i in range(2, len(labels)):
                result += math.log(self.skip_transition_probs[labels[i]][labels[i - 2]])

            # All skip-emission probabilities
            for i in range(1, len(sentence)):
                word, previous_label = sentence[i], labels[i - 1]
                if word in self.skip_emissions.keys():
                    result += math.log(self.skip_emission_probs[word][previous_label])
            
            return result
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        # For each sentence
        for sentence in data:
            words = sentence[0]
            tags = sentence[1]

            # For each word in sentence
            for i, word in enumerate(words):
                # Add word to vocabulary with it's count classified by it's tag
                # (The same word can appear as different tag)
                if word not in self.vocabulary.keys():
                    self.vocabulary[word] = dict()
                
                if tags[i] not in self.vocabulary[word].keys():
                    self.vocabulary[word][tags[i]] = 0
                self.vocabulary[word][tags[i]] += 1

                # Add count for each tag
                if tags[i] not in self.tag_counts.keys():
                    self.tag_counts[tags[i]] = 0
                self.tag_counts[tags[i]] += 1

                # Calculate transition counts between tags
                if i > 0:
                    current_tag, previous_tag = tags[i], tags[i - 1]
                    if current_tag not in self.transitions.keys():
                        self.transitions[current_tag] = dict()
                    
                    if previous_tag not in self.transitions[current_tag].keys():
                        self.transitions[current_tag][previous_tag] = 0
                    self.transitions[current_tag][previous_tag] += 1

                # Calculate skip transitions
                if i > 1:
                    current_tag, pre_previous_tag = tags[i], tags[i - 2]
                    if current_tag not in self.skip_transitions.keys():
                        self.skip_transitions[current_tag] = dict()

                    if pre_previous_tag not in self.skip_transitions[current_tag].keys():
                        self.skip_transitions[current_tag][pre_previous_tag] = 0
                    self.skip_transitions[current_tag][pre_previous_tag] += 1

                # Calculate skip emissions
                if i > 0:
                    current_word, previous_tag = word, tags[i - 1]
                    if current_word not in self.skip_emissions.keys():
                        self.skip_emissions[current_word] = dict()

                    if previous_tag not in self.skip_emissions[current_word].keys():
                        self.skip_emissions[current_word][previous_tag] = 0
                    self.skip_emissions[current_word][previous_tag] += 1


        total_words = sum(self.tag_counts.values())
        for tag, count in self.tag_counts.items():
            self.priors[tag] = count / total_words

        # Get the list of unique tags (should be =12)
        self.tags = [t for t in self.tag_counts.keys()]

        # Set tag with the largest prior probability
        self.max_prior = max(self.priors, key=self.priors.get)
        
        # Calculate transition probabilities
        for current_tag in self.tags:
            self.transition_probs[current_tag] = dict()
            for previous_tag in self.tags:
                if previous_tag in self.transitions[current_tag].keys():
                    self.transition_probs[current_tag][previous_tag] = (self.transitions[current_tag][previous_tag] + self.alpha) / self.tag_counts[previous_tag]
                else:
                   self.transition_probs[current_tag][previous_tag] = self.alpha / self.tag_counts[previous_tag]

        # Calculate skip transition probs 
        for current_tag in self.tags:
            self.skip_transition_probs[current_tag] = dict()
            for pre_previous_tag in self.tags:
                if pre_previous_tag in self.skip_transitions[current_tag].keys():
                    self.skip_transition_probs[current_tag][pre_previous_tag] = (self.skip_transitions[current_tag][pre_previous_tag] + self.alpha) / self.tag_counts[pre_previous_tag]
                else:
                   self.skip_transition_probs[current_tag][pre_previous_tag] = self.alpha / self.tag_counts[pre_previous_tag]
    
        # Calculate emission probs
        for word in self.vocabulary.keys():
            self.emission_probs[word] = dict()
            for label in self.tags:
                if label in self.vocabulary[word].keys():
                    self.emission_probs[word][label] = (self.vocabulary[word][label] + self.alpha) / self.tag_counts[label]
                else:
                    self.emission_probs[word][label] = self.alpha / self.tag_counts[label]

        # Calculate skip emission probs
        for word in self.skip_emissions.keys():
            self.skip_emission_probs[word] = dict()
            for tag in self.tags:
                if tag in self.skip_emissions[word].keys():
                    self.skip_emission_probs[word][tag] = (self.skip_emissions[word][tag] + self.alpha) / self.tag_counts[tag]
                else:
                    self.skip_emission_probs[word][tag] = self.alpha / self.tag_counts[tag]

        # print('Tags:', self.tags, "\n")
        # print('Transitions:', self.transitions, "\n")
        # print('Transition probs:', self.transition_probs, "\n")
        # print('Vocab:', list(self.vocabulary.items())[: 5], "\n")
        # print('Emission probs:', list(self.emission_probs.items())[: 5], "\n\n")
        # print('Skip Transitions:', self.skip_transitions, "\n")
        # print('Skip Transition probs:', self.skip_transition_probs, "\n")
        # print('Skip Emissions:', list(self.skip_emissions.items())[: 5], "\n")
        # print('Skip Emission probs:', list(self.skip_emission_probs.items())[: 5], "\n")


    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        output = []
        # For each word in sentence
        for word in sentence:
            posteriors = {}
            for tag in self.tags:
                if word in self.vocabulary.keys():
                    posteriors[tag] = self.emission_probs[word][tag] * self.priors[tag]
                else:
                    # if novel word, just take a decision based on the priors
                    posteriors[tag] = self.priors[tag]

            # Get the tag with the max prob
            max_tag = max(posteriors, key=posteriors.get)
            output.append(max_tag)

        self.simple_output = output
        return output
        # return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        '''
        Method to implement the Viterbi algorithm using
        Initial distribution: self.priors ; Transition probabilities: self.transition_probs; &
        Emission probabilities: self.emission_probs
        '''

        n = len(sentence)
        output = [None for _ in range(n)]
        viterbi_table = [[None for _ in range(n)] for _ in range(len(self.tags))]

        # Fill 1st column of table
        # If 1st word in vocab
        word0 = sentence[0]
        for i in range(len(self.tags)):
            current_tag = self.tags[i]
            if word0 in self.vocabulary.keys():
                viterbi_table[i][0] = [math.log(self.emission_probs[word0][current_tag]) + math.log(self.priors[current_tag]), -1]
            else:
                # if novel word, no emission prob available
                viterbi_table[i][0] = [math.log(self.priors[current_tag]), -1]

        # For all other columns
        for j in range(1, n):
            word = sentence[j]
            for i in range(len(self.tags)):
                current_tag = self.tags[i]
                if word in self.vocabulary.keys():
                    e = self.emission_probs[word][current_tag]
                else:
                    # If novel word, decsion would be taken just based on the transition probs down below
                    e = 1
                
                x = []
                for k in range(len(self.tags)):
                    previous_tag = self.tags[k]
                    x.append(math.exp(viterbi_table[k][j - 1][0]) * self.transition_probs[current_tag][previous_tag])
                viterbi_table[i][j] = [math.log(e) + math.log(max(x)), x.index(max(x))]
                    

        # print(f'\nVT: {viterbi_table}\n\n\n')
        # for i in range(len(self.tags)):
        #     for j in range(n):
        #         print(round(viterbi_table[i][j][0], 3), viterbi_table[i][j][1], end=' ')
        #     print("\n")

        # After table is filled, backtrack from the last column to find the path
        next_i = None
        for j in range(n - 1, -1, -1):
            if j == n - 1:
                # Get row number with maximum value in that column
                values = [viterbi_table[i][j] for i in range(len(self.tags))]
                max_i = values.index(max(values, key=lambda x: x[0]))
                # print(self.tags[max_i])
                output[j] = self.tags[max_i]
                next_i = viterbi_table[max_i][j][1]
            else:
                output[j] = self.tags[next_i]
                # print(self.tags[next_i])
                next_i = viterbi_table[next_i][j][1]

        return output
        #  return [ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        '''
        Method to predict POS tag for every word in the sentence using
        Gibbs Sampling - Markov Chain Monte Carlo method
        '''
        output = []
        samples = list()

        n  = len(sentence)
        # sample_0 = ['noun' for _ in range(n)]

        # Start with a random initialisation for the unobserved variables
        sample_0 = [random.choice(self.tags) for _ in range(n)]
        samples.append(sample_0)

        for _ in range(5000):
            curr = samples[-1]  # The most recent value
            curr_copy = [t for t in curr]

            # Pick an unobserved variable uniformly randomly
            i = random.randint(0, n - 1)
            probs = []

            # Depending on the value of i, 
            # sample from the posterior distribution using that variable's Markov blanket
            if i == n - 1:
                for tag in self.tags:
                    x = 1
                    if n == 2:
                        x *= self.transition_probs[tag][curr[i - 1]]
                    elif n >= 3:
                        x *= self.transition_probs[tag][curr[i - 1]] * self.skip_transition_probs[tag][curr[i - 2]]

                    if sentence[i] in self.emission_probs.keys():
                        x *= self.emission_probs[sentence[i]][tag]
                        
                    probs.append(x)
            elif i == n - 2:
                for tag in self.tags:
                    x = self.transition_probs[curr[i + 1]][tag] * self.priors[tag]

                    if n == 2:
                        x *= self.priors[tag]
                    elif n == 3:
                        x *= self.transition_probs[tag][curr[i - 1]]
                    elif n > 3:
                        x *= self.transition_probs[tag][curr[i - 1]] * self.skip_transition_probs[tag][curr[i - 2]]

                    if sentence[i] in self.emission_probs.keys():
                        x *= self.emission_probs[sentence[i]][tag]
                        
                    if sentence[i + 1] in self.skip_emission_probs.keys():
                        x *= self.skip_emission_probs[sentence[i + 1]][tag]

                    probs.append(x)
            elif i == 1:
                for tag in self.tags:
                    x = self.transition_probs[tag][curr[i - 1]] * self.transition_probs[curr[i + 1]][tag] * self.skip_transition_probs[curr[i + 2]][tag]

                    if sentence[i] in self.emission_probs.keys():
                        x *= self.emission_probs[sentence[i]][tag]
                            
                    if sentence[i + 1] in self.skip_emission_probs.keys():
                        x *= self.skip_emission_probs[sentence[i + 1]][tag]

                    probs.append(x)
            elif i == 0:
                for tag in self.tags:
                    x = self.priors[tag] * self.transition_probs[curr[i + 1]][tag] * self.skip_transition_probs[curr[i + 2]][tag]

                    if sentence[i] in self.emission_probs.keys():
                        x *= self.emission_probs[sentence[i]][tag]
                            
                    if sentence[i + 1] in self.skip_emission_probs.keys():
                        x *= self.skip_emission_probs[sentence[i + 1]][tag]

                    probs.append(x)
            else:
                for tag in self.tags:
                    x = self.transition_probs[tag][curr[i - 2]] * self.transition_probs[tag][curr[i - 1]] * self.transition_probs[curr[i + 1]][tag] * self.skip_transition_probs[curr[i + 2]][tag]
                    if sentence[i] in self.emission_probs.keys():
                        x *= self.emission_probs[sentence[i]][tag]   

                    if sentence[i + 1] in self.skip_emission_probs.keys():
                        x *= self.skip_emission_probs[sentence[i + 1]][tag]

                    probs.append(x)
                
            probs = [(x / sum(probs)) for x in probs]
            # print(probs, sum(probs))

            # Flip the weighted, biased 12-sided coin
            curr_copy[i] = np.random.choice(self.tags, p=probs)

            # Append the new sample in which
            # only a single variable was sampled keeping all others the same as the
            # previous one
            samples.append(curr_copy)
    
        # print(n, len(samples))
        # print(samples[10], "\n", samples[-1])

        # Estimate the best sampling for each word,
        # For each individual word, check which part of speech occurred most often
        for i in range(n):
            # Traverse the column
            col = [sample[i] for sample in samples]

            # Create a dictionary for the word to check 
            # which POS occured how many times
            d = dict.fromkeys(self.tags, 0)
            for c in col:
                d[c] += 1

            # Predict the one that occurred most often
            max_tag = max(d, key=d.get)
            output.append(max_tag)
            
        return output
        # return [ "noun" ] * len(sentence)

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

