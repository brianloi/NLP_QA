'''
Author: Brian Loi
Date: May 1, 2019
Course: ISTA355
Final Project

This file contains all the functions used for my ISTA355 Final Project. 
The purpose of the file is to accomplish the task of incorporating question
answering features to a classifier in order to replicate a open ended question
answer model or search engine. For more information, please read the write-up.
'''
import os
import pandas as pd
import numpy as np
import nltk
import math
import progressbar
import string
from collections import Counter
from operator import itemgetter
import gc
import random
import pickle

def WikiData_to_DF():
    '''
    This function loads the Wikipedia DataFrame and dictionary with words as its keys 
    mapped to the number of pages they appear on from pickle files created beforehand
    and returns these two objects.

    Parameter: None

    Returns: a Wikipedia DataFrame containining the titles, text, and weighed vectors
        of wikipedia pages, and dictionary with words as its keys mapped to the number
        of pages they appear on
    '''
    with open('total_dict.pickle', 'rb') as handle:
        total_dict = pickle.load(handle)
    with open('Wiki_DF.pickle', 'rb') as handle:
        output = pickle.load(handle)

    return output, total_dict

def create_wikiData(txt_data_dir):
    '''
    This function reads through a given directory, its subdirectories and files,
    reading each file and parsing the file into a large Wikipedia DataFrame, with
    the page titles as its index, and having columns for the textual content of the
    page along with its weighed (TF-IDF) vectors of that textual content. A dictionary
    with keys that are words mapped to the number of pages the words appear on is also
    created to help with the TF-IDF weighing. The dataframe and dictionary are saved into
    pickle files.

    Parameters:
        txt_data_dir - a string representing the file path to the parsed
            Wikipedia page dump directory

    Returns: None
    '''
    if (os.path.isdir(txt_data_dir)) == False:
        print("Error: parsed data file not found or incorrect filename")

    print("*Extracting WikiPages into a DataFrame")
    wiki_file_dirs = []
    for walk in os.walk(txt_data_dir): # os.walk() returns (dir, subdirs, filenames)
        for filename in walk[2]:
            wiki_file_dirs.append(walk[0]+'/'+filename)
    
    wiki_file_dirs.pop(0)   #remove DS_store directory

    index = []
    cols = ["text_content"]
    data = []
    for file in wiki_file_dirs:
        in_file = open(file, 'r')
        for line in in_file:
            if line[:4] == "<doc":
                doc_info = line.split('\"')
                #url = doc_info[3]
                title = doc_info[5]
                index.append(title)
                temp = [""]
            elif line[:5] == "</doc":
                data.append(temp)
            else:
                temp[0] += line

        in_file.close()

    output = pd.DataFrame(data, index=index, columns=cols)
    print("*Creating vector dicts for each page")

    total_dict = {}

    vects = []
    for index in output.index:
        txt = output.loc[index, 'text_content']
        txt_dict = calc_vec_dict(txt)

        for key in txt_dict:
            if key not in total_dict:
                total_dict[key] = 1
            else:
                total_dict[key] += 1
        vects.append(str(txt_dict))

    print("*Weighing Vectors (TF-IDF)")

    n_docs = len(vects)
    for i in range(len(vects)):
        txt_dict = eval(vects[i])
        result = {}
        total = sum(txt_dict.values())
        for key in txt_dict:
            #TF(L-logarithm)
            tf = np.log( 1+(txt_dict[key]/total)  )   # L-logarithm
            #IDF(T)
            num_word = total_dict[key]
            idf = np.log(n_docs/num_word)   # num of docs / num of docs that word appears in
            #TF-IDF
            result[key] = tf*idf
        vects[i] = str(result)

    output['vect_dict'] = pd.Series(vects, index=output.index)

    if os.path.exists("Wiki_DF.pickle"):
        os.remove("Wiki_DF.pickle")
    if os.path.exists("total_dict.pickle"):
        os.remove("total_dict.pickle")

    with open('Wiki_DF.pickle', 'wb') as handle:
        pickle.dump(output, handle)
    with open('total_dict.pickle', 'wb') as handle:
        pickle.dump(output, handle)

    return None

def calc_vec_dict(a_str):
    '''
    This function takes a string, and turns the string into a dictionary in which
    the keys are the words of the string and maps to the number of occurences of
    that word in the string. The dictionary represents a vector. Punctuation is
    excluded from the "words" and quotes will be replaced to avoid eval() errors
    in future.

    Parameters:
        a_str - a string representing some textual content

    Returns: a dictionary with the words of the strings as the key mapped to the
        number of occurences of that word in the string
    '''
    result = dict(Counter(a_str.lower().replace("\'", "").split()))
    for punct in string.punctuation:
        if punct in result:
            result.pop(punct)
    return result

def dict_str_to_dict(d_str):
    '''
    This function uses the eval() function to convert a string
    representing a dictionary object into the dictionary object
    and returns that dictionary.

    Parameters:
        d_str - a dictionary in the form and type of a string

    Returns: a dictionary represented from the string
    '''
    return eval(d_str)

def extend_vect_dicts(dict1, dict2):
    '''
    This function "extends" two vector dictionaries so that their lengths
    are the same; both dictionaries contain the same keys. The function
    returns these extended dictionaries.

    Parameters:
        dict1 - a dictionary representing a vector of a text
        dict2 - a dictionary representing another vector of a text

    Returns: two dictionaries that have been "extended" so that they contain
        the same keys
    '''
    vect_dict1 = dict1.copy()
    vect_dict2 = dict2.copy()

    #all_keys = set( list(dict1.keys()) + list(dict2.keys()) )
    all_keys = set().union(list(dict1.keys()), list(dict2.keys()) )

    for key in all_keys:
        if key not in vect_dict1:
            vect_dict1[key] = 0
        if key not in vect_dict2:
            vect_dict2[key] = 0
    return vect_dict1, vect_dict2

def calc_cosine_sim(vect_arr1, vect_arr2):
    '''
    This function takes in two arrays and calculates its dot product 
    in order to find its cosine similarity, and returns that value.

    Parameters:
        vect_arr1 - an array representing a text's vector 
        vect_arr2 - an array representing a text's vector 

    Returns: The cosine similarity (dot product) of the two vector arrays
        passed in
    '''
    return np.dot(vect_arr1, vect_arr2)

def norm(a_dict):
    '''
    This function takes a vector dictionary that has been extended and
    turns the values of the dictionary into an array representing the
    vector normalized, and returns that array.

    Parameters:
        a_dict - a vector dictionary that has been extended already

    Returns: an array representing a normalized vector
    '''
    #After extended
    s_lst = sorted(a_dict.items(), key=itemgetter(0))
    v_lst = list(map(itemgetter(1), s_lst))

    arr = np.array(v_lst)
    del v_lst
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return ( arr / norm )

def TF_IDF(a_dict, n_docs, total_dict):
    '''
    This function weighs the argument vector dictionary according to TF-IDF(LTN) and
    returns the newly weighed vector dictionary. This function should only be used for 
    queries or question strings.

    Parameters:
        a_dict - a vector dictionary with words as keys mapped to the number of times they
            occured in some string
        n_docs - an int representing the total number of documents (Wiki pages)
        total_dict - a dicitonary with words as keys mapping to the number of documents
            that those words appear in

    Return: a dictionary with its values weighed with TF-IDF(LTN)
    '''
    result = {}
    #Iterate through the dict, and calculate TF-IDF for each word's value
    total = sum(a_dict.values())
    #Convert the dict into a list (sorted)
    v_lst = []
    for key in sorted(a_dict):
        if a_dict[key] == 0:
            result[key] = 0
            v_lst.append(0)
            continue
        #TF(L-logarithm)
        tf = np.log( 1+(a_dict[key]/total)  )   # L-logarithm

        #IDF(T)
        if key not in total_dict:
            idf = 1
        else:
            num_word = total_dict[key]
            idf = np.log(n_docs/num_word)   # num of docs / num of docs that word appears in

        #TF-IDF
        result[key] = tf*idf

    return result

def get_qa_data(csv_file):
    '''
    This function opens up a csv file, passed in as a string, and
    reads the file into a DataFrame. From the DataFrame, the questions
    and answers are extracted into tuples that are added to a list. That
    list is returned.

    Parameters:
        csv_file - a string representing a csv file

    Returns: a list of tuples representing a list of questions and answers
    '''
    question_answers = []
    df = pd.read_csv(csv_file)

    for row in df.index:
        question = df.loc[row, 'Question']
        answer = df.loc[row, 'Answer']
        question_answers.append( (question, answer) )

    return question_answers

def clean_extract_qa_data(csv_file, new_file):
    '''
    This function converts the entire QA (Jeopardy) csv file into a DataFrame,
    and reads through the DataFrame, storing questions and answers with answers that
    are in the Wikipedia dump into a list. That list is then shuffled and split into
    training, dev-test, and test sets with a 60-20-20 percent ratio. The function then
    stores each of these sets into its own csv file.

    Parameters:
        csv_file - a string representing the entire QA set (Jeopardy)
        new_file -  a string representing the directory/file prefix for new set files

    Returns: None
    '''
    wiki_df, total_dict = WikiData_to_DF()
    question_answers = []
    df = pd.read_csv(csv_file)
    
    for row in df.index:
        question = df.loc[row, 'Question']
        answer = df.loc[row, 'Answer']
        if answer in wiki_df.index:
            if '\'' in question or '\"' in question or 'href' in question:
                continue
            question_answers.append( (question, answer) )

    print('Total QAs: '+str(len(question_answers)))
    random.shuffle(question_answers)
    split1 = int(len(question_answers)*0.6)
    split2 = int(len(question_answers)*0.8)

    trainSet = question_answers[:split1] 
    devTest = question_answers[split1:split2]
    testSet = question_answers[split2:]

    print("Training Set Size: "+ str(len(trainSet)))
    print("DevTest Set Size: " + str(len(devTest)))
    print("Test Set Size: "+ str(len(testSet)))

    #Remove any existing files
    if os.path.exists(new_file+"_train.csv"):
        os.remove(new_file+"_train.csv")
    if os.path.exists(new_file+"_devTest.csv"):
        os.remove(new_file+"_devTest.csv")
    if os.path.exists(new_file+"_test.csv"):
        os.remove(new_file+"_test.csv")

    #Convert sets to DFs
    df_train = pd.DataFrame(trainSet, columns = ['Question', 'Answer'])
    df_devTest = pd.DataFrame(devTest, columns = ['Question', 'Answer'])
    df_test = pd.DataFrame(testSet, columns = ['Question', 'Answer'])
    #Write out DFs as CSVs
    df_train.to_csv(new_file+"_train.csv", encoding='utf-8')
    df_devTest.to_csv(new_file+"_devTest.csv", encoding='utf-8')
    df_test.to_csv(new_file+"_test.csv", encoding='utf-8')

    print("*QA Data Extraction Successful*")
    print()

    return None

def eval_qa(n):
    '''
    This function loads in the QA data and Wikipedia dump data and goes through the
    devTest QA set, evaluating each question through mean reciprocal ranks. 
    If the answer is not within the top n answers, then the score for that question is zero.
    The total of these scores is accumulated and printed out to show the overall 
    performance or accuracy of the model (solely cosine similarity).
    
    Parameters:
        n - an int representing the length of the top answers list; how many possible answers
            are kept track when looking for answers

    Returns: None
    '''
    wiki_df, total_dict = WikiData_to_DF()
    print("*WikiPage DataFrame Completed")
    print("*Extracting Question and Answer Dataset")
    qa_lst = get_qa_data("./data/QA/QA_data_devTest.csv")
    #qa_lst = pd.read_csv("./data/QA/QA_data_devTest.csv")
    print("*Questions and Answers Extraction Completed")
    print("*Testing Model Against QA Set...")

    #LIMITER:
    qa_lst = qa_lst[:30]

    score = 0
    total_score = 0

    #Search Progress Bar
    bar = progressbar.ProgressBar(maxval=len(qa_lst), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    count = 0

    for qa in qa_lst:
        query = qa[0].lower()
        answer = qa[1]
        
        sim_lst = []
        q_dict = calc_vec_dict(query)
        #Conduct TF-IDF ON QUERY
        q_v_dict = TF_IDF(q_dict, len(wiki_df.index), total_dict)

        for index in wiki_df.index:
            q_vect_dict = q_v_dict
            #Extract the Wiki Page's Text Vector Dict
            t_dict = dict_str_to_dict(wiki_df.loc[index, "vect_dict"])

            #Extend the dictionaries so that they are the same size
            q_vect_dict, t_vect_dict = extend_vect_dicts(q_vect_dict, t_dict)

            #Normalize Both
            t_norm_vect = norm(t_vect_dict)
            q_norm_vect = norm(q_vect_dict)

            #Calculate Cosine Similarity
            sim = calc_cosine_sim(q_norm_vect, t_norm_vect)

            sim_lst.append( (index,sim) )

        #Update Search Progress
        count += 1
        bar.update(count)

        top_n = sorted(sim_lst, key=itemgetter(1), reverse=True)[:11]
        del sim_lst
        answer_lst = list(map( itemgetter(0),top_n ))

        #Evaluate answer results.
        if answer in answer_lst:
            score += ( 1/(answer_lst.index(answer)+1) )
        else:
            total_score += n

        del top_n
        del q_norm_vect
        del t_norm_vect

    performance = score/total_score
    print()
    print("Performance: "+ str(performance))

def ask_question_mode():
    '''
    This function loads in the wikipedia dump data and allows the user to input their 
    own question (accepts any string, no constraints). The model then runs cosine similarity
    checks with all of the Wikipedia pages and returns the top 10 cosine similar results (as
    the Wikipedia page titles along with the similarity value). The program will keep asking the
    user for input until 'quit' is entered as the question. This function is primarily used
    for testing the model for optimization, errors, etc.

    Parameters: None
    Returns: None
    '''
    wiki_df, total_dict = WikiData_to_DF()
    print("*WikiPage DataFrame Completed")
    print("Ask a question or enter quit to exit the program")
    print()

    while(True):
        print()
        question = input("Question: ").lower()
        if (question == 'quit'):
            break
        elif (question == "show_df"):
            print(wiki_df)
            print(len(wiki_df.index))
        elif (question == "show_pages"):
            for idx in wiki_df.index:
                print(idx)
        else:
            #Tracker vars:
            sim_lst = []  # This is a list of tuples (Page Title, Cosine Sim Value)

            # Get question
            q_dict = calc_vec_dict(question)
            #Conduct TF-IDF ON QUERY
            q_v_dict = TF_IDF(q_dict, len(wiki_df.index), total_dict)

            bar = progressbar.ProgressBar(maxval=len(wiki_df.index), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            count = 0

            for index in wiki_df.index:
                q_vect_dict = q_v_dict
                #Extract the Wiki Page's Text Vector Dict
                t_dict = dict_str_to_dict(wiki_df.loc[index, "vect_dict"])
                #Extend the dictionaries so that they are thes same size
                q_vect_dict, t_vect_dict = extend_vect_dicts(q_vect_dict, t_dict)
                #Normalize Both
                t_norm_vect = norm(t_vect_dict)
                q_norm_vect = norm(q_vect_dict)

                #Calculate Cosine Similarty
                sim = calc_cosine_sim(q_norm_vect, t_norm_vect)
                #Update search progress
                count += 1
                bar.update(count)

                if sim > 0:
                    sim_lst.append( (index, sim) )

            #Print Top 5 Answers (Page Titles)
            top_n = sorted(sim_lst, key=itemgetter(1), reverse=True)[:10]
            del sim_lst
            print()
            for i in range(1,len(top_n)+1):
                print(str(i) + ": "+ str(top_n[i-1][0]) + ", "+ str(top_n[i-1][1]))

            gc.collect()

def get_cosine_sim(question, wiki_df, total_dict, answer):
    '''
    This function finds and calculates the cosine similarity between the
    answer's Wikipedia page's textual content through its vector form and
    the vector of the question being asked. The function returns that cosine
    similarity value.

    Parameters:
        question - a string representing a question
        wiki_df - a DataFrame containing Wikipedia page info
        total_dict - a dictionary of keys that are words mapped to the number
            of Wikipedia pages that they appear in
        answer - a string representing the answer to the question

    Returns: a float representing the cosine similarity of two vectors (that represent text)
    '''
    # Get vector of quesetion
    q_dict = calc_vec_dict(question)
    #Conduct TF-IDF ON QUERY
    q_vect_dict = TF_IDF(q_dict, len(wiki_df.index), total_dict)

    #Extract the Wiki Page's Text Vector Dict
    t_dict = dict_str_to_dict(wiki_df.loc[answer, "vect_dict"])
    #Extend the dictionaries so that they are thes same size
    q_vect_dict, t_vect_dict = extend_vect_dicts(q_vect_dict, t_dict)
    #Normalize Both
    t_norm_vect = norm(t_vect_dict)
    q_norm_vect = norm(q_vect_dict)

    #Calculate Cosine Similarty
    sim = calc_cosine_sim(q_norm_vect, t_norm_vect)

    return sim

def document_features(wiki_df, total_dict, question, answer, negative=False):
    '''
    This function creates a feature dictionary to be used in a classifier and
    returns that dictionary.

    Parameters:
        wiki_df - a DataFrame containing Wikipedia page info
        total_dict - a dictionary of keys that are words mapped to the number
            of Wikipedia pages that they appear in
        question - a string representing a question
        answer - a string representing an answer to the question
        negative - a bool determining whether or not an inccorect answer is to be
            used for the features (to train the model on false answers)

    Returns: a dictionary of features
    '''
    if negative:
        new_a = answer
        while(new_a == answer):
            new_a = wiki_df.index[random.randint(0, len(wiki_df.index)-1)]
        answer = new_a

    features = {}
    features['cosine_sim'] = get_cosine_sim(question, wiki_df, total_dict, answer)

    bigrams = set([b for b in nltk.bigrams(question.split())])
    features['contains_bigram'] = False
    for b in bigrams:
        if ("".join(b) in wiki_df.loc[answer, 'text_content']):
            features['contains_bigram'] = True

    trigrams = set([t for t in nltk.trigrams(question.split())])
    features['contains_trigram'] = False
    for t in trigrams:
        if ("".join(t) in wiki_df.loc[answer, 'text_content']):
            features['contains_trigram'] = True

    features['question_word_len'] = len(question.split())
    features['doc_word_len'] = len(wiki_df.loc[answer, 'text_content'].split())

    return features

def classifier():
    '''
    This function loads in the QA data sets (the train, devTest, and test sets) and
    creates feature sets based on all of these sets. For the trainSet, a negative feature
    set is also created to provide the classifier with negative cases. After all the feature
    sets are generated, a naive Bayes Classifier is trained and its performance is evaluated
    on the devTest set and the 10 most informative features are shown. In the end, the function
    will ask the user if they want to also test with the test set. If yes, then the same classifier
    will be used to test against the test set.

    Parameters: None
    Returns: None
    '''
    wiki_df, total_dict = WikiData_to_DF()
    print("*WikiPage DataFrame Completed")
    #load in QA data
    trainSet = get_qa_data("./data/QA/QA_data_train.csv")
    devTest =  get_qa_data("./data/QA/QA_data_devTest.csv")
    testSet = get_qa_data("./data/QA/QA_data_test.csv")
    #Create Feature Sets
    print("*Creating Feature Sets*")
    print("This may take 3-5 minutes. Please wait...")
    featureSets = [ (document_features(wiki_df, total_dict, question, answer), 'correct') for (question, answer) in trainSet]
    neg_featureSets = [ (document_features(wiki_df, total_dict, question, answer, negative=True), 'wrong') for (question, answer) in trainSet]
    training = featureSets + neg_featureSets

    dev_TestSets = [ (document_features(wiki_df, total_dict, question, answer), 'correct') for (question, answer) in devTest]
    print("*Feature Sets Completed*")
    #train model
    print("*Training Classifier")
    classifier = nltk.NaiveBayesClassifier.train(training)
    #Evaluate Performance
    performance = nltk.classify.accuracy(classifier, dev_TestSets)
    print()
    print("Performance on DevTest Set: " + str(performance))
    print()
    classifier.show_most_informative_features(10)
    print()
    user = input("Would you like to use the Classifier on the test set? (y/n) ")

    if user.lower() == 'y':
        print("*Loading Features for Test Set...")
        testSet_features = [ (document_features(wiki_df, total_dict, question, answer), 'correct') for (question, answer) in testSet]
        performance = nltk.classify.accuracy(classifier, testSet_features)
        print()
        print("Performance on Test Set: " + str(performance))
        print()
    return

def main():
    '''
    This program asks the user for a command. Based on the command, the program will
    either (0) Run a classifier to be trained and evaluated, (1) evaluate the QA model, 
    (2) ask the model a question through user input, (3) extract and split the QA data 
    into train, devTest, test set files., and (4) Create a Wikipedia DataFrame containing
    all the information we care about and save it into a pickle file. If the command
    is not recognized, the program closes.

    Notes:
    (0) - Trains a classifier and tests it on the dev-test set
        and possibly the test set (this is the main option)
    (1) - Currently, this take an extremely long time without limiting the number of questions
        yields poor performance
    (2) - takes user input. Primarily used for program testing
    (3) - The program closes after this is command is run.
    (4) - Creates and saves the Wiki DF as a pickle file
    '''
    cont = True

    while (cont):
        print("What would you like to do?")
        print("\t0 - Classifier")
        print("\t1 - Evaluate Question Answering (Cosine-Sim)")
        print("\t2 - Ask the QA Model Questions")
        print("\t3 - Extract and Split QA Data")
        print("\t4 - Create Wiki-Data")
        print("\tq - Quit")
        cmd = input("Enter Command Number: ")
        if cmd.strip() == "1":
            eval_qa(10)
            cont = False
        elif cmd.strip() == "2":
            ask_question_mode()
            cont= False
        elif cmd.strip() == "3":
            clean_extract_qa_data("./data/QA/JEOPARDY.csv", "./data/QA/QA_data")
        elif cmd.strip() == "4":
            create_wikiData("./parsed_data")
            print("*WikiData Pickles Completed*")
            print()
        elif cmd.strip() == "0":
            classifier()
            cont = False
        elif cmd.strip() == "q":
            return
        else:
            print("Invalid Command. Exiting...")  
            cont = False


if __name__ == '__main__':
    main()


