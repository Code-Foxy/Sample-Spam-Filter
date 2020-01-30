import email
import os
import math

def load_tokens(email_path):
    
    msg = email.message_from_file(open(email_path, encoding = 'utf8'))
    it = email.iterators.body_line_iterator(msg)
    tokens = []

    #for each line in the given iterator, extend tokens with its tokens
    for line in it:
        tokens.extend(line.split())
        
    return tokens
    

def log_probs(email_paths, smoothing):
    
    tokens = []
    prob = {}

    #create an entry for unknown tokens with count of zero
    prob['<UNK>'] = 0

    #create a token list from all the email paths
    for email_path in email_paths:
        tokens.extend(load_tokens(email_path))

    #for all tokens count all of their occurances and store in the dictionary
    for token in tokens:
        if token not in prob:
            prob[token] = 1 
        else:
            prob[token] += 1
            
    denom = len(tokens) + (smoothing * len(prob))

    #change the word counts in the dictionary to their Laplace-smoothed log probabilities
    for key in prob:
        prob[key] = math.log( (prob[key] + smoothing) / denom )

    return prob

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):

        spam_paths = [os.path.join(spam_dir, f) for f in os.listdir(spam_dir)]
        ham_paths = [os.path.join(ham_dir, f) for f in os.listdir(ham_dir)]
        denom = (len(spam_paths) + len(ham_paths))
        
        self.spam_dict = log_probs(spam_paths, smoothing)
        self.ham_dict = log_probs(ham_paths, smoothing)
        
        self.p_spam = math.log(len(spam_paths) / denom)
        self.p_ham = math.log(len(ham_paths) / denom)
    
    def is_spam(self, email_path):
        
        tokens = []
        mail_dict = {}

        spam_prod, ham_prod = 0, 0

        #retrieve all tokens and their counts for the given path
        tokens.extend(load_tokens(email_path))
        for token in tokens:
            if token not in mail_dict:
                mail_dict[token] = 1
            else:
                mail_dict[token] += 1

        #compute the intermediate products (sums in log space)
        for key in mail_dict:
            
            if key in self.spam_dict:
                spam_prod += (self.spam_dict[key]*mail_dict[key])
            else:
                spam_prod += (self.spam_dict['<UNK>']*mail_dict[key])

            if key in self.ham_dict:
                ham_prod += (self.ham_dict[key]*mail_dict[key])
            else:
                ham_prod += (self.ham_dict['<UNK>']*mail_dict[key])

        #based on probabilities, determine if the given email is spam
        if self.p_spam + spam_prod > self.p_ham + ham_prod:
            return True
        else:
            return False

    def most_indicative_spam(self, n):
        
        list_out = []

        #find all tokens present in both ham and spam
        for key in self.spam_dict:
            if key in self.ham_dict:
                #I convert out of log form to do the addition and then convert into log form once more
                p_w = math.log(math.exp(self.p_spam) * math.exp(self.spam_dict[key]) + math.exp(self.p_ham) * math.exp(self.ham_dict[key]))
                list_out.append((self.spam_dict[key] - p_w , key))

        #given the list of all values, we want to sort in decending order so we reverse sort
        list_out.sort(reverse = True)

        #return the first n values of the list returning only the token and not the value
        return [i[1] for i in list_out[:n]]
        

    def most_indicative_ham(self, n):

        #this function is almost identical to the last except the ham_dict is being searched and we refer to
        #the ham dict value in the numerator
        list_out = []
        
        for key in self.ham_dict:
            if key in self.spam_dict:
                p_w = math.log(math.exp(self.p_spam) * math.exp(self.spam_dict[key]) + math.exp(self.p_ham) * math.exp(self.ham_dict[key]))
                list_out.append((self.ham_dict[key] - p_w , key))

        list_out.sort(reverse = True)

        return [i[1] for i in list_out[:n]]
