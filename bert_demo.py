'''
Created on 07.09.2020

@author: Sarah Boening
Basierend auf dem Tutorial: https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/

Frag BERT - Demo fuer ein fertig trainiertes Netzwerk
'''

import torch 
from transformers import BertForQuestionAnswering, BertTokenizer # benoetigte Klassen
import re # regex bibliothek

if __name__ == '__main__':
    '''
    1. FERTIGES MODELL LADEN
    '''
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') # Text braucht einen Tokenizer, der die Woerter in erlaubte Tokens teilt
    
    '''
    2. BENOETIGTE DATEN 
    '''
    #question = "What is the size of BERT-Large?"
    question = "What is the embedding size?"
    answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34 GB, so expect it to take a couple minutes to download to your Colab instance."
    #question = "What is the input data?"
    #answer_text = "As input data, the machine was powered from an ideal current source with a maximum amplitude of 15 A at a rotor speed of 1000 rpm to obtain a mechanical power of 75 kW."

    '''
    3. DATEN VORBEREITEN
    '''
    # Tokenizer auf Daten anwenden, Netzwerk braucht numerischen Input 
    # Text -> Tokens -> IDs    
    input_ids = tokenizer.encode(question, answer_text)

    print('The input has a total of {:} tokens.'.format(len(input_ids)))
    
    # Das ganze rueckwaerts
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # Suche `[SEP]` Token
    sep_index = input_ids.index(tokenizer.sep_token_id)
    # Die Frage ist Frage + SEP
    num_seg_a = sep_index + 1
    # Der Rest ist der Text mit der Anwort
    num_seg_b = len(input_ids) - num_seg_a
    # Teil 1 = 0, Teil 2 = 1 zur Unterscheidung
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    '''
    4. MODELL LAUFEN LASSEN
    '''
    # Beispiel -> Netzwerk
    start_scores, end_scores = model(torch.tensor([input_ids]), 
                                     token_type_ids=torch.tensor([segment_ids])) 
    # start_scores = Anfang der Antwort, end_scores = Ende der Antwort
    # Finde wahrscheinlichste Positionen ( groesste Wahrscheinlichkeit)
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    
    # Von Tokens zurueck zu Text
    answer = ' '.join(tokens[answer_start:answer_end+1])
    answer = answer.replace("##", '')
    answer = re.sub(r' +', '', answer)
    # Ausgabe des Ergebnisses
    print("Question: " + question)
    print("Text: " + answer_text)
    print('Answer: ' + answer + '')
