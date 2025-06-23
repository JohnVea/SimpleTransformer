# from tokenizer import Tokenizer
import random
import math
# tokenizer = Tokenizer(vocabulary="low lowest John the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Wilson James John Robert Michael David William Richard Joseph Mary Patricia Jennifer Linda Elizabeth Barbara Susan Margaret\n Over hill, over dale, Thorough bush, thorough brier, Over park, over pale, Thorough flood, thorough fire! I do wander everywhere, Swifter than the moon's sphere; And I serve the Fairy Queen, To dew her orbs upon the green; The cowslips tall her pensioners be; In their gold coats spots you see; Those be rubies, fairy favours; In those freckles live their savours; I must go seek some dewdrops here, And hang a pearl in every cowslip's ear.", vocabSize=713)
# lowwjohn = tokenizer.encode("low w john is cool and amazing <PAD>")
#print("Decoded: " + tokenizer.decode(lowwjohn))

class Transformer:
    def __init__(self, input_dim,learning_rate=0.001):
        self.inputVector = [0 for _ in range(input_dim)]
        self.outputVector = [0 for _ in range(input_dim)]
        self.learning_rate = learning_rate
        self.w_Q = [[random.uniform(-1, 1)* math.sqrt(1/input_dim) for _ in range(input_dim)] for _ in range(input_dim)] # Query matrix
        self.w_K = [[random.uniform(-1, 1)* math.sqrt(1/input_dim) for _ in range(input_dim)] for _ in range(input_dim)] # Key matrix
        self.w_V = [[random.uniform(-1, 1)* math.sqrt(1/input_dim) for _ in range(input_dim)] for _ in range(input_dim)]  # Value matrix
        self.attention_W = [[0 for _ in range(input_dim)] for _ in range(input_dim)]

    def forward(self, input_matrix):
        sequence_length = batch_size = len(self.inputVector)
        key = [[0 for _ in range(sequence_length)] for _ in range(batch_size)]
        value = [[0 for _ in range(sequence_length)] for _ in range(batch_size)]
        for a in range(batch_size):
            for b in range(sequence_length):
                key[a][b] = self.w_K[b][a] * input_matrix[b]
                value[a][b] = self.w_V[b][a] * input_matrix[b]
        for i in range(batch_size):
            query = [[0 for _ in range(len(self.w_Q))] for _ in range(len(self.w_K))]
            attention_scores = [[[0 for _ in range(len(self.w_Q))] for _ in range(len(self.w_K))] for _ in range(batch_size)]
            input_vector = input_matrix[i]
            for c in range(len(self.w_K)):
                for d in range(len(self.w_Q)):
                    query[c][d] = self.w_Q[c][d] * input_vector
            for c in range(len(self.w_K)):
                for d in range(len(self.w_Q)):
                    attention_scores[c][d] = query[d][c] * key[c][d]
            sqrt_k = math.sqrt(len(query[0]))
            for e in range(len(self.w_K)):
                for f in range(len(self.w_Q)):
                    self.attention_W[e][f] = attention_scores[e][f] / sqrt_k

       
        for l in range(len(self.outputVector)):
            sum = 0
            for m in range(len(self.outputVector)):
                sum += self.attention_W[l][m] * value[l][m]
                # print("Sum: ", sum)
            self.outputVector[l] = sum
        return self.outputVector
    
    def transpose_matrix(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = [[0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return transposed



# will accept a 1 dimensional array of numbers
# and return a 1 dimensional array of numbers, same lenght as the input
arr = [random.uniform(-1, 1)* math.sqrt(1/10)  for _ in range(3)]
print("Array: ", len(arr))
print(arr)

# arb = [2, 3]
# for i in range(len(arr)):
#     for j in range(len(arr[0])):
#         arr[i][j] = arr[i][j] * arb[j]
# print("Modified Array: ", arr)
transformer = Transformer(input_dim=len(arr))
output = transformer.forward(arr)
print("Output: ", output)



# def transpose_matrix(matrix):
#     rows = len(matrix)
#     cols = len(matrix[0])
#     transposed = [[0 for _ in range(rows)] for _ in range(cols)]
#     for i in range(rows):
#         for j in range(cols):
#             transposed[j][i] = matrix[i][j]
#     return transposed

# matrix = [[1, 2], [4, 5], [7, 8]]
# transposed_matrix = transpose_matrix(matrix)
# print(transposed_matrix) # Output: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
# a = [[1] [2] [3]]
# print(transposed_matrix * a)