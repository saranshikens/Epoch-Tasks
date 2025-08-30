import numpy as np

class BahdanauAttention:
    def __init__(self, W1, W2, v):
        self.W1 = W1
        self.W2 = W2
        self.v = v

    def forward(self, key, query):
        scores = np.tanh(key @ self.W1.T + query @ self.W2.T)
        att_scores = scores @ self.v
        att_weights = self.softmax(att_scores)
        context_vector = att_weights @ key

        return att_weights, context_vector
    
    def softmax(self, X):
        X = np.exp(X - np.max(X))
        return X/np.sum(X)
    



class LuongDotAttention:
    def forward(self, key, query):
        att_scores = key @ query
        att_weights = self.softmax(att_scores)
        context_vector = att_weights @ key

        return att_weights, context_vector
    
    def softmax(self, X):
        X = np.exp(X - np.max(X))
        return X/np.sum(X)
    



class LuongGeneralAttention:
    def __init__(self, W):
        self.W = W

    def forward(self, key, query):
        att_scores = (key @ self.W.T) @ query
        att_weights = self.softmax(att_scores)
        context_vector = att_weights @ key

        return att_weights, context_vector
    
    def softmax(self, X):
        X = np.exp(X - np.max(X))
        return X/np.sum(X)
    



class LuongConcatAttention:
    def __init__(self, W, v):
        self.W = W
        self.v = v

    def forward(self, key, query):
        query = np.tile(query, (key.shape,1))
        concat_matrix = np.concatenate([key, query], axis=1)
        scores = np.tanh(concat_matrix @ self.W.T)
        att_scores = scores @ self.v
        att_weights = self.softmax(att_scores)
        context_vector = att_weights @ key

        return att_weights, context_vector
    
    def softmax(self, X):
        X = np.exp(X - np.max(X))
        return X/np.sum(X)
    


