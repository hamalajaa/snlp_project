# Predictive typing assistance using recurrent neural networks                                                                                                                                                                                                                                                                                          
Group project for the Statistical Natural Language Processing course.

## Instructions

### To train the model:

1. This project recommends to use virtualenv. Install requirements from requirements.txt

2. Specify the path to a tokenized training file where each sentence is on its own line and words are separated by spaces. This can be done in main.py by modifying the value of variable `data_file`.

3. Run the project with `python main.py`

The program will run and required model files will be created.

### To predict and test the model:

1. Specify the path to the trained model and vocabulary information. For example `model_load_path = "./results/24.326k_800_800/model.pth"` and `vocab_info_load_path = "./results/24.326k_800_800/vocab.json"`

2. In `main.py`, set `main(load=True)`

2. Run the project with `python main.py`
