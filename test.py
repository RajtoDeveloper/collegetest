1.BFS
def bfs(graph, start):
 visited = []
 queue = [start]
 while queue:
    node = queue.pop(0)
    if node not in visited:
        print(node,end=" ")
        visited.append(node)
        queue.extend(graph[node])
graph = {
 'A': ['B', 'C'],
 'B': ['D', 'E'],
 'C': ['F'],
 'D': [],
 'E': ['F'],
 'F': []
}
print("BFS Traversal:")
bfs(graph, 'A') 

2.DFS
def dfs(graph, node, visited=None):
 if visited is None:
    visited = []
 if node not in visited:
    print(node, end=" ")
    visited.append(node)
    for neighbor in graph[node]:
        dfs(graph,neighbor,visited)
graph = {
 'A': ['B', 'C'],
 'B': ['D', 'E'],
 'C': ['F'],
 'D': [],
 'E': ['F'],
 'F': []
}
print("DFS Traversal:")
dfs(graph, 'A') 

3.tic-tac-toe
print("------------N-DIMENSIONAL TIC TAC TOE game by guru 99.com------------")
def show(board):
    for row in board:
        print(" ".join(row))
    print()
while True:
    board = [['_', '_', '_'],['_', '_', '_'],['_', '_', '_']]
    show(board)
    turn = 'X'
    moves = 0
    while True:
        print("Player ", turn)
        r = int(input("Enter row (0-2): "))
        c = int(input("Enter col (0-2): "))
        if board[r][c] != '_':
            print("Already taken! Try again.")
            continue
        board[r][c] = turn
        moves += 1
        show(board)
        win = False
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == turn:
                win = True
            if board[0][i] == board[1][i] == board[2][i] == turn:
                win = True
        if board[0][0] == board[1][1] == board[2][2] == turn:
            win = True
        if board[0][2] == board[1][1] == board[2][0] == turn:
            win = True
        if win:
            print("Game Over. Player", turn, "wins!")
            print("Guru99.com tic tac toe game")
            break
        if moves == 9:
            print("Game over. Players have tied the match.")
            print("Guru99.com tic tac toe game")
            break
        turn = 'O' if turn == 'X' else 'X'
    again = input("Play Guru99 tic tac toe_Game again? (y/n):").lower()
    if again != 'y':
        print("Guru99 TicTacToe game ended.")
        break

4. 8 puzzle
from collections import deque
def solve(b):
 s = sum(b, [])
 if s == list(range(9)): return 0
 m = [[1,3], [0,2,4], [1,5], [0,4,6], [1,3,5,7], [2,4,8], [3,7],[4,6,8], [5,7]]
 q = deque([(s, 0)])
 v = set()

 while q:
    t, c = q.popleft()
    if str(t) in v: continue
    v.add(str(t))
    z = t.index(0)
    for i in m[z]:
        n = t[:]
        n[z], n[i] = n[i], n[z]
        if n == list(range(9)): return c + 1
        q.append((n, c + 1))
 return -1
print(solve([[3,1,2], [4,7,5], [6,8,0]])) 

5.Water jug 
print("Water Jug problem")
x = int(input("Enter X:")) #0
y = int(input("Enter Y:"))  #0
while True:
 rno = int(input("Enter the rule no:")) #2 9 2 7 5 9 
 if rno == 1:
    if x < 4:
        x = 4
 if rno == 2:
    if y < 3:
        y = 3
 if rno == 5:
    if x > 0:
        x = 0
 if rno == 6:
    if y > 0:
        y = 0
 if rno == 7:
    if x + y >= 4 and y > 0:
        x, y = 4, y - (4 - x)
 if rno == 8:
    if x + y >= 3 and x > 0:
        x, y = x - (3 - y), 3
 if rno == 9:
    if x + y <= 4 and y > 0:
        x, y = x + y, 0
 if rno == 10:
    if x + y <= 3 and x > 0:
        x, y = 0, x + y
 print("x =", x)
 print("y =", y)
 if x == 2:
    print("The result is a goal state")
    break 

6.Travelling salesman problem
import random
from itertools import permutations
def tsp(distances):
    n = len(distances)
    best_route = []
    best_cost = float('inf')
    for route in permutations(range(n)):
        cost = 0
        for i in range(n - 1):
            cost += distances[route[i]][route[i + 1]]
        cost += distances[route[-1]][route[0]]
        if cost < best_cost:
            best_cost = cost
            best_route = route
    return best_route, best_cost
distances = [
    [0, 10, 15, 20],
    [10, 0, 30, 5],
    [15, 30, 0, 25],
    [20, 5, 25, 0]
]
route, cost = tsp(distances)
print("The best route is:", route)
print("The total cost is:", cost)

7.Tower of hanoi
def tower_of_hanoi(n, source, aux, dest):
 if n == 0:
    return
 tower_of_hanoi(n - 1, source, dest, aux)
 print(f"Move disk {n} from source {source} to destination {dest}")
 tower_of_hanoi(n - 1, aux, source, dest)
n=3
tower_of_hanoi(n,"A","B","C")

8.Monkey banana
def monkey_banana_problem(n):
 climb = 0
 bananas = 0
 hungry = True
 for i in range(n):
    if hungry:
        climb += 1
        bananas += 1
        hungry = False
    else:
        climb += 1
 return climb, bananas
n = 10
climb, bananas = monkey_banana_problem(n)
print(f"The monkey made {climb} climbs and get {bananas} bananas.")

9.Alpha beta pruning
MAX, MIN = 1000, -1000
def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]
    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best
values = [3, 5, 6, 9, 1, 2, 0, -1]
print("The optimal value is:", minimax(0, 0, True, values, MIN, MAX))

10. 8-Queens
global N
N = 4
def printSolution(board):
    for i in range(N):
        for j in range(N):
            print(board[i][j], end=' ')
        print()
def isSafe(board, row, col):
    for i in range(col):
        if board[row][i] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    return True
def solveNQUtil(board, col):
    if col >= N:
        return True
    for i in range(N):
        if isSafe(board, i, col):
            board[i][col] = 1
            if solveNQUtil(board, col + 1):
                return True
            board[i][col] = 0  
    return False
def solveNQ():
    board = [[0 for _ in range(N)] for _ in range(N)]
    if not solveNQUtil(board, 0):
        print("Solution does not exist")
        return False
    printSolution(board)
    return True
solveNQ()
            
11.Chatbot
import random
responses=["Hello,how can I help you?",
 "What do you want to talk about?",
 "I'm not sure what you mean.",
 "Can you repeat that?",
 "I'm sorry,I don't understand."]
def get_response():
 return random.choice(responses)
def start_chatbot():
 print("Hello,I'm a chatbot.What do you want to talk about?")
 while True:
    user_input=input("You:")
    if user_input.lower() in ["bye","exit","quit"]:
        print("Chatbot:Goodbye!")
        break
    response=get_response()
    print(response)
start_chatbot() 

12.Hangman
import random
words = ["apple", "orange"]
word = random.choice(words)
attempt = 10
guessed = ['_'] * len(word)
print("Welcome to Hangman")
print(" ".join(guessed))
while attempt > 0 and '_' in guessed:
    guess = input("Guess a letter: ").lower()
    if guess in word:
        for i in range(len(word)):
            if word[i] == guess:
                guessed[i] = guess
        print("Good guess:", " ".join(guessed))
    else:
        attempt -= 1
        print("Wrong! You have", attempt, "attempts left")
if '_' not in guessed:
    print("You won! The word is", word)
else:
    print("You lost! The word is", word)

13.NLTK
create 2 txt files
file1.txt - This is a simple example to remove stop words using NLTK.
file2.txt - output automatically dispalyed
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
input_path = r"C:\Users\HP\Desktop\file1.txt" # Path of your input file
output_path = r"C:\Users\HP\Desktop\file2.txt" # Path of output file
with open(input_path, "r") as f1, open(output_path, "w") as f2:
    for line in f1:
        words = line.split()
        filtered = [w for w in words if w.lower() not in stop]
        f2.write(" ".join(filtered) + "\n")
print(f"Done! Output saved to {output_path}")
