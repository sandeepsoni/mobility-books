def wordpiece_boundaries (tokenizer, 
                          tokens, 
                          start_index, 
                          end_index):
    """ Returns the start and end index of wordpieces of interest.

    Parameters
    ==========
    tokenizer (BertTokenizer): Tokenizer used to tokenize the tokens into wordpieces.
    tokens (list): The text is represented as a sequence of tokens
    start_index (int): The index (or position) of the first token of the entity of interest
    end_index (int): The index (or position) of the last token of the entity of interest
        
    Returns
    =======
    start_wordpiece (int): The index (or position) of the first wordpiece of the entity of interest
    end_wordpiece - 1(int): The index (or position) of the last wordpiece of the entity of interest
    """

    prefix_tokens = tokenizer (" ".join (tokens[0:end_index+1]))
    entity_tokens = tokenizer (" ".join (tokens[start_index:end_index+1]))
    start_wordpiece = len (prefix_tokens['input_ids'][1:-1]) - \
	                  len (entity_tokens['input_ids'][1:-1])
    end_wordpiece = len (prefix_tokens['input_ids'][1:-1])    
    return start_wordpiece, end_wordpiece-1

def tokens2wordpieces (tokenizer,
                       tokens,
		               per_entity_start, 
                       per_entity_end, 
                       loc_entity_start, 
                       loc_entity_end):

    """ Preprocess and convert tokens to wordpiece sequence

    tokenizer (BERTTokenizer): Tokenizer used to tokenize the tokens into wordpieces.
    tokens (list): The text is represented as a sequence of tokens
    per_entity_start (int): The index of the start token for the character.
    per_entity_end (int): The index of the end token for the character.
    loc_entity_start (int): The index of the start token for the location.
    loc_entity_end (int): The index of the end token for the location.
    """
    # We'll find the start and the end wordpiece for both the person and location entities
    per_wp_start, per_wp_end = wordpiece_boundaries (tokenizer, tokens, per_entity_start, per_entity_end)
    loc_wp_start, loc_wp_end = wordpiece_boundaries (tokenizer, tokens, loc_entity_start, loc_entity_end)
 
    # We'll restrict reading up to end of sentence after the last entity
    end = max (per_entity_end, loc_entity_end)
    index = len (tokens) - 1
    for index in range (end, len (tokens)):
        if tokens[index] == ".": # period
            break
                
    return index, (per_wp_start, per_wp_end), (loc_wp_start, loc_wp_end)