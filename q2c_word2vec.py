#!/usr/bin/env python
import random

import numpy as np

from helpers.utils import normalize_rows, sigmoid, get_negative_samples
from q2a_softmax import softmax
from q2b_gradcheck import gradcheck_naive


def naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset
):
    v_c = center_word_vec
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    center_word_vec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outside_word_idx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outside_vectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_center_vec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    grad_outside_vecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE
    #raise NotImplementedError


    #scores = np.matmul(U_row_mat, v_c)
    #print(center_word_vec)
    scores = np.matmul(outside_vectors, center_word_vec)
    
    """
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / np.sum(exp_scores)
    mechane = softmax_probs[0]
    mone = np.exp(np.dot(u_o.T, v_c))
    
    loss_before_log = mone/mechane
    """
    softmax_probs = softmax(scores)
    #loss = -np.log(loss_before_log)
    loss = -np.log(softmax_probs[outside_word_idx])

    #for grad_center_vec

    u_o = outside_vectors[outside_word_idx]
    
    dot_pro = np.matmul(outside_vectors, v_c)
    exp_values = np.exp(dot_pro)

    sum_exp_values = np.sum(exp_values)

    item1 = 1 / sum_exp_values

    # Calculate the matrix-vector product


    #item2 = np.sum(res, axis=0)
    calc1 = np.dot(np.exp(np.dot(outside_vectors[0], v_c)), outside_vectors[0])
    item2 = np.exp(np.dot(outside_vectors[0], v_c)) * outside_vectors[0]
    #print(item2)
    for idx, u_k in enumerate(outside_vectors):
        if idx != 0:
            item2+=np.exp(np.dot(u_k, v_c)) * u_k

    grad_center_vec =  (item1 * item2)-u_o 


    #grad_outside_vecs = -np.outer(softmax_probs, v_c)
    v_c = center_word_vec

    grad_outside_vecs = []
    for idx, u_w in enumerate(outside_vectors):
        if idx == outside_word_idx:
            grad_outside_vecs.append(-v_c + np.dot(softmax_probs[idx], v_c))
        else:
            grad_outside_vecs.append(np.dot(softmax_probs[idx], v_c))
    grad_outside_vecs = np.array(grad_outside_vecs)

### END YOUR CODE
    
    
    return loss, grad_center_vec, grad_outside_vecs


def neg_sampling_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset,
        K=10
):
  
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a center_word_vec
    and a outside_word_idx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    helper_mat = outside_vectors[neg_sample_word_indices, :]

    u_o = outside_vectors[outside_word_idx]
    v_c = center_word_vec

    dot_product = -np.dot(helper_mat, v_c)
    sig_values = sigmoid(dot_product)
    sig_log_values = np.log(sig_values)

    #for loss
    first_item = -np.log(sigmoid(np.dot(u_o.T, v_c)))
    second_item = np.sum(sig_log_values)
    loss = first_item - second_item

    #grad_center_vec
    
    dot_product = np.dot(u_o.T, v_c)
    sig_values = sigmoid(dot_product)
    #r =  -1*(1-sig_values)
    #first_item2 = r*u_o


    #in_sum = np.dot((1-sigmoid(-dot_product)), helper_mat)
    #second_item2 = np.sum(in_sum)

    my_sum1 = -1*(1-sigmoid(np.dot(u_o.T, v_c)))*u_o
    my_sum2 = (1-sigmoid(-np.dot(outside_vectors[neg_sample_word_indices][0].T, v_c)))*outside_vectors[neg_sample_word_indices][0]
    first = True
    for idx in neg_sample_word_indices:
        if first == True:
            first = False
        else:
            u_k = outside_vectors[idx]
            my_sum2+=(1-sigmoid(-np.dot(u_k.T, v_c)))*u_k    

    grad_center_vec = my_sum1 + my_sum2

    

    #grad_outside_vecs
    grad_outside_vecs = np.zeros(outside_vectors.shape)
    grad_outside_vecs[outside_word_idx] = (-(1-sig_values)*v_c)
    for index in neg_sample_word_indices:
        u_k = outside_vectors[index]
        dot_product = -np.dot(u_k.T, v_c)
        sigmoid_value = sigmoid(dot_product)
        insert = (1-sigmoid_value)*v_c
        grad_outside_vecs[index] += insert

        

    ### YOUR CODE HERE



    #raise NotImplementedError
    ### END YOUR CODE
    #print(grad_outside_vecs)

    return loss, grad_center_vec, grad_outside_vecs


def skipgram(current_center_word, outside_words, word2ind,
             center_word_vectors, outside_vectors, dataset,
             word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_center_word -- a string of the current center word
    window_size -- integer, context window size
    outside_words -- list of no more than 2*window_size strings, the outside words
    word2ind -- a dictionary that maps words to their indices in
              the word vector list
    center_word_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outside_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vec_loss_and_gradient -- the loss and gradient function for
                               a prediction vector given the outside_word_idx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    grad_center_vecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    grad_outside_vectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """
    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    center_id = word2ind[current_center_word]
    v_c = center_word_vectors[center_id]
    #v_c = current_center_word
    for word in outside_words:
        index = word2ind[word]
        temp_loss, temp_grad_center_vecs, temp_grad_outside_vectors =word2vec_loss_and_gradient(v_c, index, outside_vectors, dataset)
        loss+=temp_loss
        grad_center_vecs[center_id]+=temp_grad_center_vecs
        grad_outside_vectors+=temp_grad_outside_vectors



    return loss, grad_center_vecs, grad_outside_vectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset,
                         window_size, word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    #print(word_vectors)
    #print("end")
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2):, :]
    for i in range(batchsize):
        window_size1 = random.randint(1, window_size)
        center_word, context = dataset.getRandomContext(window_size1)

        c, gin, gout = word2vec_model(
            center_word, context, word2ind, center_word_vectors,
            outside_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec_basic():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummy_sample_token_idx():
        return random.randint(0, 4)

    def get_random_context(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
            [tokens[random.randint(0, 4)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummy_sample_token_idx
    dataset.getRandomContext = get_random_context

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient),
        dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    print("==== Gradient check for skip-gram with neg_sampling_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_loss_and_gradient),
                    dummy_vectors, "neg_sampling_loss_and_gradient Gradient")

    print("\n=== Results ===")
    print("Skip-Gram with naive_softmax_loss_and_gradient")

    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                      dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
        )
    )

    print("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print("Skip-Gram with neg_sampling_loss_and_gradient")
    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :],
                  dataset, neg_sampling_loss_and_gradient)
        )
    )
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

def tester_naive_softmax_loss_and_gradient():
    # Generate some example data for testing
    vocab_size = 10
    embedding_dim = 5

    center_word_vec = np.random.rand(embedding_dim)
    outside_word_idx = 3  # Index of the outside word
    outside_vectors = np.random.rand(embedding_dim, vocab_size)

    # Compute the loss and gradients
    loss, grad_center_vec, grad_outside_vecs = naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        None  # You can pass a dataset here if needed
    )

    # Print the results
    print("Loss:", loss)
    print("Gradient with respect to center word vector:", grad_center_vec)
    print("Gradients with respect to outside word vectors:")
    for i, gradient in enumerate(grad_outside_vecs):
        print(f"Gradient for outside word {i}:", gradient)

# Test the function
#tester_naive_softmax_loss_and_gradient()

if __name__ == "__main__":
    test_word2vec_basic()
    #tester_naive_softmax_loss_and_gradient()