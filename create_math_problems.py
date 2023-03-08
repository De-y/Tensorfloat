import os
import math
import random

# set the random seed for consistency
random_seed = 42
random.seed(random_seed)

# create the directory if it doesn't already exist
if not os.path.exists("./math"):
    os.makedirs("./math")

# define the number of equations to generate
num_equations = 1000

# generate random equations and write the result to a file
for i in range(num_equations):
    # generate a random value between 1 and num_equations
    random_num = random.randint(1, 360)
    
    # define the equations to choose from
    equations = [
        (f"sqrt({random_num})", math.sqrt(random_num)),
        (f"tan({random_num})", math.tan(random_num)),
        (f"cos({random_num})", math.cos(random_num)),
        (f"sin({random_num})", math.sin(random_num)),
    ]
    
    # choose a random equation from the list
    equation, result = random.choice(equations)
    
    # generate a random input value
    input_value = random.uniform(-10, 10)
    
    # solve the equation at the input value
    if isinstance(result, float):
        answer = result
    else:
        answer = result(input_value)
    
    # generate a random filename
    filename = f"equation_{i}.txt"
    
    # write the equation and answer to a file
    with open(f"./math/{filename}", "w") as file:
        # file.write(f"The input value is: {input_value}\n")
        file.write(f"({equation}) = {answer}")

# with input {input_value} is: 