from tqdm import tqdm
import torch
torch.manual_seed (96)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from .relation_prediction_utils import wordpiece_boundaries, tokens2wordpieces
import logging

class BERTRelationPrediction (nn.Module):
    """ Relation prediction for a pair of entities"""
    def __init__ (self,
		          model_name="bert-base-cased",
		          dims=768,
		          n_labels=8,
                  labels=[""],
		          device="cpu",
		          lr=1e-5):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained (model_name,
                                                        do_lower_case=False,
                                                        do_basic_tokenize=False)
        self.bert = BertModel.from_pretrained (model_name)
        self.n_labels = n_labels
        self.fc = nn.Linear (2*dims, self.n_labels)
        self.device = device
        self.to (self.device)
        self.optimizer = optim.Adam (self.parameters (), lr=lr)
        self.cross_entropy = nn.CrossEntropyLoss ()
        self.overall_loss = 0.0
        self.num_epochs = 0
        self.labels = labels
		
    def forward (self,
                 encoded_input, 
                 per_wp_start, 
                 per_wp_end, 
                 loc_wp_start, 
                 loc_wp_end):
        """ Constructs the forward computation graph and returns the forward computation value.
        
        encoded_input (dict): Contains the wordpieces encoded into numeric ids
        per_wp_start (int): The index of the start token for the character.
        per_wp_end (int): The index of the end token for the character.
        loc_wp_start (int): The index of the start token for the location.
        loc_wp_end (int): The index of the end token for the location.
        """

        encoded_input.to(self.device)
        _, pooled_inputs, sequence_outputs =  self.bert (**encoded_input, 
                                                         output_hidden_states=True, 
                                                         return_dict=False)
        last_layer_output = sequence_outputs[-1][0]
        per_entity_repr = last_layer_output[per_wp_start:per_wp_end+1].mean(dim=0)
        loc_entity_repr = last_layer_output[loc_wp_start:loc_wp_end+1].mean(dim=0)
        input_repr = torch.cat ((per_entity_repr, loc_entity_repr), 0)
        output = self.fc (input_repr)
        return output
    
    def load_data (self,
		           filepath,
                   *args,
                   **kwargs):
        """ Load data from filepath.
        
		filepath (str): The path of the pickle file that contains the entire training dataset.
		args: variable length arguments
		kwargs: variable length keyword arguments
		"""
        with open (filepath, "rb") as fin:
            annotations = pickle.load (fin)
        self.full_df, self.train_df, self.test_df = kwargs["preprocess"] (annotations, *args, **kwargs)
	
    def __train__ (self, 
                   text_field="context_100",
                   label_field="Valid Relation",
                   max_model_length=512,
		           sep=" "):
        # Train
        self.train()
        self.overall_loss = 0.0
        for i in tqdm (range (len (self.train_df))):
            # get the extracted quantities
            text = self.train_df[text_field].iloc[i]
            label = self.train_df[label_field].iloc[i]
            tokens = text.split (sep)
            index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end) = tokens2wordpieces (self.tokenizer,
                                                                                               tokens,
											                                                   self.train_df.iloc[i]["persons_start"],
											                                                   self.train_df.iloc[i]["persons_end"],
											                                                   self.train_df.iloc[i]["locations_start"],
											                                                   self.train_df.iloc[i]["locations_end"])
            encoded_input = self.tokenizer (sep.join (tokens[0:index+1]), return_tensors="pt")
            if len (encoded_input['input_ids'][0]) > max_model_length:
                continue
	    
            y_pred = self.forward (encoded_input, 
                                   per_wp_start, 
				                   per_wp_end,
				                   loc_wp_start,
				                   loc_wp_end)
	    
            y_truth = self.labels.index (label)
            loss = self.cross_entropy (y_pred.unsqueeze (0), torch.tensor ([y_truth]).to(self.device))
            self.optimizer.zero_grad ()
            self.overall_loss += loss.item ()
            loss.backward ()
            self.optimizer.step ()
        return self.overall_loss/len (self.train_df)
	
    def __eval__ (self, 
		          text_field="context_100",
                  label_field="Valid Relation",
		          max_model_length=512,
		          sep=" "):
        # Evaluation
        groundtruth, predictions = list (), list ()
        self.eval ()
        with torch.no_grad ():
            for i in tqdm (range (len (self.test_df))):
                text = self.test_df[text_field].iloc[i]
                label = self.test_df[label_field].iloc[i]
                tokens = text.split (sep)

                index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end) = tokens2wordpieces (self.tokenizer,
                                                                                                   tokens, 
												                                                   self.test_df.iloc[i]["persons_start"],
												                                                   self.test_df.iloc[i]["persons_end"],
												                                                   self.test_df.iloc[i]["locations_start"], 
												                                                   self.test_df.iloc[i]["locations_end"])

                encoded_input = self.tokenizer (sep.join (tokens[0:index+1]), return_tensors="pt")
                if len(encoded_input['input_ids'][0]) > max_model_length:
                    continue

                y_pred = self.forward (encoded_input, 
                                       per_wp_start, 
                                       per_wp_end,
                                       loc_wp_start, 
                                       loc_wp_end)
            
                y_truth = self.labels.index (label)
                groundtruth.append (y_truth)
                predictions.append (torch.argmax (torch.nn.functional.softmax (y_pred)).item())

        return groundtruth, predictions
				
    def start_training (self,
                        num_epochs=10, 
			            verbose=False,
                        text_field="context_100",
                        label_field="Valid Relation",
                        max_model_length=512,
	                    eval_freq_in_epochs=1):
        """  Train the model with examples and evaluate the model as we train.
        num_epochs (int): The number of epochs to train (default: 10)
        text_field (str): The field that contains the text (default: "context_100")
        label_field (str): The field that contains the label (default: "Valid Relation")
        verbose (bool):  Print debugging  and update messages if this flag is True (default: False)
        context_field (str): The column name which contains the textual context (default: "context_10")
        max_model_length (int): The maximum length in terms of wordpieces that the model can handle (default: 512)
        eval_freq_in_epochs (int): The number of epochs after which one round of evaluation is done (default: 1)
		"""
        for epoch in range(num_epochs):
            if verbose: logging.info (f"Starting Epoch {epoch+1}")
            loss = self.__train__ (text_field=text_field,
				                   label_field=label_field, 
				                   max_model_length=max_model_length)
            if verbose: logging.info (f"Loss in epoch {epoch+1}={loss}")
            if (epoch + 1) % eval_freq_in_epochs == 0:
                groundtruth, predictions = self.__eval__ (text_field=text_field,
							                              label_field=label_field,
							                              max_model_length=max_model_length)
                self.predictions = predictions
                self.groundtruth = groundtruth
            if verbose: logging.info (classification_report (groundtruth, predictions))
            self.num_epochs += 1
	
    def apply_book (self,
		            book_df,
		            text_field="context_100",
		            max_model_length=512,
		            sep=" "):
        self.eval()
        predictions = list ()
        with torch.no_grad ():
            for i in range (len (book_df)):
                text = book_df[text_field].iloc[i]
                tokens = text.split (sep)
                index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end) = tokens2wordpieces (self.tokenizer,
                                                                                                   tokens, 
                                                                                                   book_df.iloc[i]["persons_start"],
                                                                                                   book_df.iloc[i]["persons_end"],
                                                                                                   book_df.iloc[i]["locations_start"], 
                                                                                                   book_df.iloc[i]["locations_end"])

                encoded_input = self.tokenizer (sep.join (tokens[0:index+1]), return_tensors="pt")
                if len(encoded_input['input_ids'][0]) > max_model_length:
                    continue

                y_pred = self.forward (encoded_input, 
                                       per_wp_start, 
                                       per_wp_end,
                                       loc_wp_start, 
                                       loc_wp_end)
            
                predictions.append (torch.argmax (torch.nn.functional.softmax (y_pred)).item())
        return predictions
	    
    def save_model (self, model_path):
        torch.save({
            'epoch': self.num_epochs,
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss': self.overall_loss/len(self.train_df),
			}, model_path)
	
    def save_predictions (self, predictions_path):
        self.test_df.loc[:, "predictions"] = self.predictions
        self.test_df.loc[:, "groundtruth"] = self.groundtruth
        self.test_df.to_csv (predictions_path, sep=",", header=True, index=False)

    def save (self, 
              model_path="",
	          predictions_path=""):
        if not len (model_path) == 0:
            self.save_model (model_path)

        if not len (predictions_path) == 0:
            self.save_predictions (predictions_path)


class SpatialRelationPrediction (nn.Module):
	def __init__ (self, 
				  model_name="bert-base-cased", 
				  bert_dims=768, 
				  n_labels=8,
				  device="cpu",
				  lr=1e-5):
		super().__init__()
		self.tokenizer = BertTokenizer.from_pretrained(model_name, 
                                                       do_lower_case=False, 
                                                       do_basic_tokenize=False)
		self.bert = BertModel.from_pretrained(model_name)
		self.n_labels = n_labels
		self.fc = nn.Linear (2*bert_dims, self.n_labels)
		self.device = device
		self.to(self.device)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.cross_entropy=nn.CrossEntropyLoss()
		self.overall_loss = 0.0
		self.num_epochs = 0
		self.binary = (self.n_labels == 2)

	def __wordpiece_boundaries__ (self, 
								  tokens, 
								  start_index, 
								  end_index):
		""" Returns the start and end index of wordpieces of interest.
        
		Parameters
		==========
        
		tokens (list): The text is represented as a sequence of tokens
		start_index (int): The index (or position) of the first token of the entity of interest
		end_index (int): The index (or position) of the last token of the entity of interest
        
		Returns
		=======
		start_wordpiece (int): The index (or position) of the first wordpiece of the entity of interest
		end_wordpiece - 1(int): The index (or position) of the last wordpiece of the entity of interest
		"""
		prefix_tokens = self.tokenizer (" ".join (tokens[0:end_index+1]))
		entity_tokens = self.tokenizer (" ".join (tokens[start_index:end_index+1]))
		start_wordpiece = len (prefix_tokens['input_ids'][1:-1]) - \
                          len (entity_tokens['input_ids'][1:-1])
		end_wordpiece = len (prefix_tokens['input_ids'][1:-1])
        
		return start_wordpiece, end_wordpiece-1

	def __preprocess__ (self,
						tokens,
						per_entity_start, 
						per_entity_end, 
						loc_entity_start, 
						loc_entity_end):
		""" Preprocess and convert tokens to wordpiece sequence
        
		tokens (list): The text is represented as a sequence of tokens
		per_entity_start (int): The index of the start token for the character.
		per_entity_end (int): The index of the end token for the character.
		loc_entity_start (int): The index of the start token for the location.
		loc_entity_end (int): The index of the end token for the location.
        """

        # We'll find the start and the end wordpiece for both the person and location entities
		per_wp_start, per_wp_end = self.__wordpiece_boundaries__ (tokens, per_entity_start, per_entity_end)
		loc_wp_start, loc_wp_end = self.__wordpiece_boundaries__ (tokens, loc_entity_start, loc_entity_end)
 
		# We'll restrict reading up to end of sentence after the last entity
		end = max (per_entity_end, loc_entity_end)
		index = len (tokens) - 1
		for index in range (end, len (tokens)):
			if tokens[index] == ".": # period
				break
                
		return index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end)

	def forward (self,
				 encoded_input, 
				 per_wp_start, 
				 per_wp_end, 
				 loc_wp_start, 
				 loc_wp_end):
		""" Constructs the forward computation graph and returns the forward computation value.
        
        encoded_input (dict): Contains the wordpieces encoded into numeric ids
        per_wp_start (int): The index of the start token for the character.
        per_wp_end (int): The index of the end token for the character.
        loc_wp_start (int): The index of the start token for the location.
        loc_wp_end (int): The index of the end token for the location.
        """

		encoded_input.to(self.device)
		_, pooled_inputs, sequence_outputs =  self.bert (**encoded_input, 
                                                         output_hidden_states=True, 
                                                         return_dict=False)
		last_layer_output = sequence_outputs[-1][0]
		per_entity_repr = last_layer_output[per_wp_start:per_wp_end+1].mean(dim=0)
		loc_entity_repr = last_layer_output[loc_wp_start:loc_wp_end+1].mean(dim=0)
		input_repr = torch.cat ((per_entity_repr, loc_entity_repr), 0)
		output = self.fc (input_repr)
		return output

	def load_training_data  (self, 
							 filepath, 
							 window_size=100,
							 test_ids_file="",
							 training_frac=0.8):
		""" Load training data from filepath.
        
		filepath (str): The path of the pickle file that contains the entire training dataset.
		window_size (int): Select the window size to index (default: 100); can be one of 10, 50, 100.
		training_frac (float): The proportion of examples used for training and the rest for evaluation;
                               0  < training_frac < 1.0 (default: 0.8)
		test_ids_file (str): The path of the .txt file that contains the test ids.
        
		The side effect of this function is the creation of object attributes like 
        
		"""
		self.all_labels = [ "NO RELATIONSHIP ASSERTED",
                            "TOWARD(got there)",
                            "FROM",
                            "NEAR",
                            "IN",
                            "NEGATIVE ASSERTION",
                            "THROUGH",
                            "TOWARD (uncertain got there)",
                            "BAD LOC",
                            "BAD PER",
                            "UNCERTAIN ASSERTION"]
		
		self.spatial_labels = [ "NO RELATIONSHIP ASSERTED",
								"TOWARD(got there)",
								"FROM",
								"NEAR",
								"IN",
								"NEGATIVE ASSERTION",
								"THROUGH",
								"TOWARD (uncertain got there)",
								"UNCERTAIN ASSERTION"]
        
		self.bad_labels = [ "BAD LOC",
                            "BAD PER"]
        
		self.label_names = {0:"GOOD", 1:"BAD"}
		self.accepted_labels = ["GOOD", "BAD"]

		with open (filepath, "rb") as fin:
			annotations = pickle.load (fin)
		self.full_df = annotations[window_size]
        
		self.full_df = self.full_df[self.full_df["Spatial Relation"] != ""]
		self.full_df = self.full_df[self.full_df["Spatial Relation"].isin (self.all_labels)]
		self.full_df["Spatial SuperRelation"] = self.full_df["Spatial Relation"].isin (self.bad_labels)

		if not self.binary:
			self.full_df = self.full_df[self.full_df["Spatial Relation"].isin (self.spatial_labels)]

		if test_ids_file == "":
			self.train_df, self.test_df = train_test_split(self.full_df, 
						  								   test_size=1-training_frac, 
														   random_state=96)
		else:
			with open (test_ids_file) as fin:
				test_ids = [line for line in fin]
			self.train_df = self.full_df[~self.full_df["ID"].isin (test_ids)]
			self.test_df = self.full_df[self.full_df["ID"].isin (test_ids)]

	def start_training (self, 
						num_epochs=10, 
						verbose=False,
						context_field="context_10",
						max_model_length=512,
						eval_freq_in_epochs=1):

		"""  Start training the model with examples and evaluate the model as we train.
        
		num_epochs (int): The number of epochs to train (default: 10)
		verbose (bool):  Print debugging  and update messages if this flag is True (default: False)
		context_field (str): The column name which contains the textual context (default: "context_10")
		max_model_length (int): The maximum length in terms of wordpieces that the model can handle (default: 512)
		eval_freq_in_epochs (int): The number of epochs after which one round of evaluation is done (default: 1)

		"""
		for epoch in range(num_epochs):
			if verbose: print (f"Epoch: {epoch+1}")
			self.overall_loss = 0.0
			self.train()
			# Train
			for i in tqdm (range (len (self.train_df))):
				# get the extracted quantities
				text = self.train_df[context_field].iloc[i]
				label = self.label_names[int (self.train_df["Spatial SuperRelation"].iloc[i])] if self.binary else self.train_df["Spatial Relation"].iloc[i]
				tokens = text.split (" ")
				index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end) = self.__preprocess__ (tokens, 
																									 self.train_df.iloc[i]["persons_start"],
																									 self.train_df.iloc[i]["persons_end"], 
																									 self.train_df.iloc[i]["locations_start"], 
																									 self.train_df.iloc[i]["locations_end"])
				encoded_input = self.tokenizer (" ".join (tokens[0:index+1]), return_tensors="pt")
				if len(encoded_input['input_ids'][0]) > max_model_length:
					continue
                    
				y_pred = self.forward (encoded_input, 
									   per_wp_start, 
									   per_wp_end,
									   loc_wp_start,
									   loc_wp_end)
                
				y_truth = self.accepted_labels.index (label) if self.binary else self.spatial_labels.index (label)
				loss = self.cross_entropy (y_pred.unsqueeze (0), torch.tensor ([y_truth]).to(self.device))
				self.optimizer.zero_grad ()
				self.overall_loss += loss.item()
				loss.backward ()
				self.optimizer.step ()
			if verbose: print (f"Train Loss: {self.overall_loss/len (self.train_df)}")
			self.num_epochs += 1

			if epoch % eval_freq_in_epochs == 0:
				groundtruth, predictions = list (), list ()
				self.eval ()
				with torch.no_grad ():
					for i in tqdm (range (len (self.test_df))):
						text = self.test_df[context_field].iloc[i]
						label = self.label_names[int (self.test_df["Spatial SuperRelation"].iloc[i])] if self.binary else self.test_df["Spatial Relation"].iloc[i]
						tokens = text.split (" ")
						index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end) = self.__preprocess__ (tokens, 
																											 self.test_df.iloc[i]["persons_start"],
																											 self.test_df.iloc[i]["persons_end"],
																											 self.test_df.iloc[i]["locations_start"], 
																											 self.test_df.iloc[i]["locations_end"])

						encoded_input = self.tokenizer (" ".join (tokens[0:index+1]), return_tensors="pt")
						if len(encoded_input['input_ids'][0]) > max_model_length:
							continue

						y_pred = self.forward (encoded_input, 
											   per_wp_start, per_wp_end,
											   loc_wp_start, loc_wp_end)
            
						y_truth = self.accepted_labels.index (label) if self.binary else self.spatial_labels.index (label)
						groundtruth.append (y_truth)
						predictions.append (torch.argmax (torch.nn.functional.softmax (y_pred)).item())

				if verbose: print (classification_report (groundtruth, predictions))
						
			

	def evaluate_book (self, 
					   book_df,
					   context_field="context_100",
					   max_model_length=512):
		self.eval()
		predictions = list ()
		with torch.no_grad ():
			for i in range (len (book_df)):
				text = book_df[context_field].iloc[i]
				tokens = text.split (" ")
				index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end) = self.__preprocess__ (tokens, 
																									 book_df.iloc[i]["persons_start"],
																									 book_df.iloc[i]["persons_end"],
																									 book_df.iloc[i]["locations_start"], 
																									 book_df.iloc[i]["locations_end"])
					
				#print (book_df.iloc[i]["persons_start"],
				#	   book_df.iloc[i]["persons_end"],
				#	   book_df.iloc[i]["locations_start"], 
				#	   book_df.iloc[i]["locations_end"])

				#print (per_wp_start, per_wp_end, loc_wp_start, loc_wp_end)

				encoded_input = self.tokenizer (" ".join (tokens[0:index+1]), return_tensors="pt")
				if len(encoded_input['input_ids'][0]) > max_model_length:
					continue

				y_pred = self.forward (encoded_input, 
									   per_wp_start, per_wp_end,
									   loc_wp_start, loc_wp_end)

				#print (torch.nn.functional.softmax (y_pred))
            
				predictions.append (torch.argmax (torch.nn.functional.softmax (y_pred)).item())

		return predictions
				
				
	def save_model (self, model_path):
		torch.save({
			'epoch': self.num_epochs,
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss': self.overall_loss/len(self.train_df),
			}, model_path)