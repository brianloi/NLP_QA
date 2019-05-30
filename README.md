# Natural Language Processing: Question Answering

This Python script program utilizes a WikiMedia dump, of 15030 Wikipedia pages, in order to perform the task of question answering using natural language processing principles. The program has different forms of open-ended question answering: free question answering based on user input, evaluation of the question answering model (takes extremely long), and measuring the most informative features and performance of the model through a naive Bayes classifier. The question answering model creates vectors for individual Wikipedia pages and calculates the cosine similarity of those vectors with the vector of the query, using a TF-IDF model. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

1. Clone the repository
2. Unzip the files found in the directory parsed_data. Make sure that the contents of the zips are in the parsed_data directory, and make sure that there is nothing else in the parsed_data folder except those contents (get rid of the zips).
3. Run the master.py file
4. Enter the command number 4, to create pickle files of the Wiki data.
5. Enter the command number 3, to create testing, training, and dev-test files for the classifier model.

After step 5, all files should be set up for the program. Command numbers 0, 1, and 2 should now be functional.
This steps only needed to be completed once, in order to generate the necessary files for the program to run.

### Prerequisites/Data

This repository contains all necessary files for this program to run.
It does not contain the Wikimedia dump in its original format (due to its extremely large size).
The program also uses a Jeopardy data set for questions/answers for the classifier model. It can be found here: https://data.world/sya/200000-jeopardy-questions.

If you want to use your own WikiMedia dump, for more updated Wikipedia pages or a different set of pages, download the dump from Wikimedia and use WikiExtractor to convert the dump file into readable raw text files.

WikiMedia dumps can be found here: https://dumps.wikimedia.org/enwiki/latest/.

WikiExtractor can be found here: http://medialab.di.unipi.it/wiki/Wikipedia_Extractor#Introduction.

## Running The Program

After all the steps in Getting Started have been completed, the program will be able to run.
The python script program runs in the terminal or powershell.

There are three modes that this program can run: Naive Bayes Classifier model, Open-ended user input question answering, and raw performance evaluation of the question answering model.

After running the python script master.py:

* Naive Bayes Classifier
```
1. Enter 0 as the command number
2. The program will train a naive Bayes Classifier model using the test set (This may take a few minutes)
3. The program will test the classifier model using the dev-test set
4. The program will print out the performance (as a decimal) and then output the top 10 more informative features in the model.
5. You will be prompted on whether or not you would like to also test that model with the testing set. Enter either y or n.
6. If you enter y, the program will then load the testing set, test the model and output the accuracy for that testing.
7. The program will then exit itself.
```

* Open-ended User Input Question Answering
```
1. Enter 2 as the command number
2. The program will continuously as for a question until quit is entered.
3. Enter a question, phrase, or words as a query
4. The program will calculate the top 10 answers and print them out, along with their cosine similarity values to the query
5. Enter another question, until you quit
```

* Raw Performance Evaluation
```
Warning: This takes an extremely long time to run, and is not suggested!
1. Enter 1 as the command number
2. The number of questions being used to evaluate the model is limited to 30. To change this, alter the eval_qa() function.
3. The program will answer those 30 questions and compare it answers to the true answers and print out a score.
4. The program will then exit itself.
```

### More Information

For more information, you may refer to the Write Up documentation found in the paperwork directory.

## Author

* **Brian Loi** - *Author* - [GitLink](https://github.com/brianloi)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/brianloi/NLP_QA/blob/master/LICENSE) file for details

