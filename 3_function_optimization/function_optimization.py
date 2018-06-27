
# coding: utf-8

# # Optimizing a neural network

# 
# Optimization techniques are a core aspect of training machine learning models, and it is useful to be aware of some of the concepts associated with optimization theory. In this notebook we'll cover some of the more important optimization concepts that we typically encounter when studying machine learning algorithms.
# 
# When we think of optimization, we might be thinking of one of two things. On the one hand, we might have a function for which we would like to find the inputs that returns the maximum or minimum value. We can think of this as function minimization/maximization. On the other hand, we might have a parameterized function for which we would like to find the set of paramters that return the smallest error between an output and some target value. We can think of this as parameter optimization/error minimization.

# #### Imports



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')




def formatter(ax=None, title=''):
    "Helper to set a grid and 0 axis lines"
    if not ax:
        plt.grid(True, alpha=0.4); plt.axvline(0, alpha=0.6); plt.axhline(0, alpha=0.6); plt.title(title)
        return plt
    else:
        ax.grid(True, alpha=0.4); ax.axvline(0, alpha=0.6); ax.axhline(0, alpha=0.6); ax.set_title(title)
        return ax


# ## Function minimization/maximization
# 
# When we define an unparameterized function, we may encounter situations where we would like to minimize or maximize the function. In otherwords, we would like to find the input that produces the maximum output. For some functions, this is trivial. For example, a linear function that increases as the input increases is maximized by selecting the input that is equivalent to the functions upper bound. 



def linear(x, m):
    assert max(x) <=  10, 'upper bound crossed'
    assert min(x) >= -10, 'lower bound crossed'
    return x * m




x = np.array([-10, 10])
y = linear(x, 2)
plt.plot(x, y, linestyle='--')
formatter()
plt.xlim((-30, 30));plt.ylim((-30, 30));


# We have bounded the min and max output of the __linear__ function above at -20 and 20, respectively. If we cross the upper bound...



x = np.array([-10, 11])
linear(x, m=2)


# ... we get an assertion error. If we don't bound this function, it is intuitively obvious that the function will increase towards infinity when we increase the input and decrease towards negative inifinity as we decrease the input.
# 
# For non-linear functions, we may want to minimize the function or maximize the function depending on the problem.
# 
# For those interested in what real world miniization/maximization problems might look like:
# https://www.math.drexel.edu/~jwd25/CALC1_SPRING_06/lectures/lecture9.html

# ### Calculus - the tool for minimization and maximization

# For many, the word 'Calculus' is a loaded word that conjures fear from the deepest darkest places of the mind. It is typically considered the point at which advanced math begins and the desire to continue studying ends. Its the point at which the world of math seems to spread infinitely to the left and right with no telling how far forward it goes.
# 
# 
#  - derivative (first order, second order)
#  - partial derivative (Jacobian, Hessian)
#  - gradient
#  - local minimum/minima
#  - local maximum/maxima
#  - critical point
#  

# ### The derivative

# An important concept from calculus is the derivative. The derivative of a function gives us information about the rate at which the output of that function changes as its inputs change. To find the derivative of a function can be tricky depending on the function, so in this notebook we'll only talk about functions whose derivatives are relatively simple to find.
# 
# For a linear function, like the one just above, there is a constant slope. For example, for the function...
# 
# $$f(x) = 2x$$
# 
# ...it should be clear if you pass a value $x$ to $f(x)$, then the output of $f(x)$ will always return a value that $ 2 * x$. In other words, there is a _constant tranformation_ to the value of x applied to the input $x$ which is indicted by the _constant_ $2$.
# 
# For a non-linear function, this is not the case. For example, the function...
# 
# $$f(x) = x^2$$
# 
# ...does not scale in a constant was as the linear function does. Instead, it scales with the square of the input. Lets take a peek at what this looks like.



def non_linear(x):
    return x ** 2

x = np.linspace(-10, 10, 100)
y = non_linear(x)
plt.plot(x, y, linestyle='--')
formatter(title='f(x)=x^2')
plt.xlim((-5, 5));plt.ylim((-2, 20));


# If you look closely, you'll see that when $x$ is near zero, the slope of the the curve at that point $x$ is pretty shallow (doesn't increase quickly), but when $x$ is far from zero, the slope of the curve is pretty steep. The slope can be visualized by imagining a line that is tangent to the function at a given point.
# 
# What does this even mean!? Lets take a look!



def slope_at_point(x):  # Don't worry just yet about why this function is
    return 2*x




fig, axes = plt.subplots(nrows=1, ncols=3, **{'figsize':(16, 6)})
x = np.linspace(-10, 10, 100)
y_nl = non_linear(x)

# Plot 1
ax = axes[0]
ax.plot(x, y_nl, linestyle='--')
formatter(ax=ax, title='f(x) = X^2')

#---------------------------------------------------------
# Plot 2
ax = axes[1]
m = slope_at_point(1)
y = linear(x, m=m)
y_shift = non_linear(1)

ax.plot(x, y_nl, linestyle='--') # original non-linear function
ax.plot(x+1, y+y_shift, linestyle='--')  # Tangent line, i.e. slope at point
ax.scatter(1, y_shift, s=30, c='black')   # point
formatter(ax=ax, title='Shallow slope')

#---------------------------------------------------------
# Plot 3
ax = axes[2]
m = slope_at_point(4)
y = linear(x, m=m)
y_shift = non_linear(4)

ax.plot(x, y_nl, linestyle='--') # original non-linear function
ax.plot(x+4, y+y_shift, linestyle='--')  # Tangent line, i.e. slope at point
ax.scatter(4, y_shift, s=30, c='black')   # point
formatter(ax=ax, title='Steep slope');


# So for all the $x$ values between 0 and 2, the slope of the orange line is more shallow that the slope of the organe line between values 2 and 4.
# 
# ##### The function that we use to draw this line is the derivative.
# 
# The the derivative is incredibly informative. In the example above, it tells us which direction we need to shift the input values in order to find the minimum output value of the function. (In other examples, it may give us information about the maximium value). 
# 
# The reason it gives us this information is that when a line curves down (like it does in the function plotted above), and then curves back up, we know that the point at which the line transitions from a downward curve to an upward curve is going to be the lowest point in the plot. We also can see that if we plot the derivative function at this point, it will have a slope of zero.



# Plot the derivative when the derivative equals zero
x = np.linspace(-10, 10, 100)
m = slope_at_point(0)
y = linear(x, m=m)
y_shift = non_linear(0)

plt.plot(x, y_nl, linestyle='--') # original non-linear function
plt.plot(x, y, linestyle='--')  # Tangent line, i.e. slope at point
plt.scatter(0, y_shift, s=30, c='black'); plt.title("f'(x); x = 0 where f(x) = x^2");


# You can see now that when the derivative function returns zero, we've hit a point at which the output of the function is minimized - since the output will increase if we go either left or right.
# 
# For functions where there is only a single point that may be considered the **global minimum**, we use the term *convex function*. $f(x) = x^2$ is one such function. There is a subfield of optimization called convex optimization that is interested is finding efficient or even analytical solutions of convex functions. 
# 
# https://en.wikipedia.org/wiki/Convex_optimization  
# https://en.wikipedia.org/wiki/Convex_function
# 
# 
# But not all functions are convex. For example, when we begin making a function  adding terms to a function (making it a polynomial) that are of , we begin to encounter functions that may have multiple **local minima**, of which one or more may be the **global minima**.
# 
# Take for example the function:
# 
# $$f(x) = \frac{1}{4}x^4 + \frac{1}{3}x^3 - \frac{1}{2}x^2$$



def higher_degree_polynomial(x):  
    return ((1/4)*x**4) + ((1/3)*x**3) -((1/2)*x**2)

x = np.linspace(-10, 10, 200)
y = higher_degree_polynomial(x)

plt.plot(x, y, linestyle='--')
formatter(title='Non convex function');
plt.xlim((-5, 5)); plt.ylim((-1.5, 1.5));


# It is probably clear from the graph above that there are two **local minima** for the function, only one of which is the global minimum. This is therefor a non-convex function for which there can be no convex-optimization technique that finds the global minimum. (However, if the function should be bounded such that, say, $x >= 0$, then in some respects the function may be considered convex). 
# 
# So to find the points at which the function above is locally minimized, we find the **derivitive function** $f'(x)$ and find the points x at which the derivative function is equal to zero i.e. $f'(x) = 0$

# ## How do you find the derivative?

# This is probably a good time to talk briefly about how to find the derivative of a function. In school (as I recall anyways), there were two ways to find the derivative of a function. One way was to take the limit of the function between two points as the distance between those two points went to zero. The other way was use a trick involving bring down the exponent. I think the trick is the method people typically remember and use whenever possible, so we'll just recap that here.
# 
# Take the function:
# $$f(x) = 4x^2 + 5$$
# 
# To find the derivative $f'(x)$, we perform the following operations:
# 
# $$f'(x) = (2)*4x^{2-1} + (0)*5$$
# 
# $$ = 8x$$
# 
# So we multiply the term by the exponent value, then subtract 1 from the exponent of the term. Constants are eliminated.
# 
# So for the function plotted above, we would do:
# 
# $$f(x) = \frac{1}{4}x^4 + \frac{1}{3}x^3 - \frac{1}{2}x^2$$
# 
# Therefore..
# 
# $$f'(x) = x^3 + x^2 - x$$
# 
# If you're interested, a quick youtube search brough up the harder way to find the derivative at a point:  
# https://www.youtube.com/watch?v=PMKr97AstNU

# So to find the points in the plot that exist as minima and maxima, we simply find the solutions for which the output of this function is equal to zero. There are various approaches to do this. In school, we learned how to do this **analytically** by factorizing the function such that we were left with factors with obvious zero solutions.
# 
# For example, to find the zeros of the function:
# $$f(x) = 2x^2 + 5x - 3$$
# 
# We factorize the function to get:
# 
# $$f(x) = (2x + 1)(x + 3)$$
# 
# Setting this to zero means that any time either group within the parenthesis is equal to zero, the function evaluates to zero. So the zeros are:
# 
# $$2x+1 = 0$$
# $$x=-\frac{1}{2}$$
# 
# and
# 
# $$x+3 = 0$$
# $$x=-3$$
# 
# When the function doesn't have integer solutions, there are other ways to find the root of the polynomials such as empoloying the rational zeros theorum. 
# 
# But this is TEDIOUS! Especially when we have to do it a million times!

# **This is one reason that derivative is so interesting. The derivative tells us not only the slope at a given input, but also the direction in which we can alter the input $x$ so as to decrease the output!**

# If we take the previous example of the convex function $f(x) = x^2$, we can see this in practice.



x = np.linspace(-10, 10, 100)
y_nl = non_linear(x)

plt.plot(x, y_nl, linestyle='--')
formatter(ax=None, title='f(x) = X^2');




x = np.linspace(-10, 10, 100)

point = 3
m = slope_at_point(point)
y = linear(x, m=m)
y_shift = non_linear(point)

plt.plot(x, y_nl, linestyle='--') # original non-linear function
plt.plot(x+point, y+y_shift, linestyle='--')  # Tangent line, i.e. slope at point
plt.scatter(point, y_shift, s=30, c='black')   # point
formatter(title='');
plt.xlim((-7, 7)); plt.ylim((-3, 20));


# The 'slope_at_point' function is defined as $f'(x) = 2x$ which is the derivative function of $f(x) = x^2$. At the input point 3, the slope is _positive_ 6.



slope_at_point(3)


# So this tells us that the curve is increasing from left to right along the x-axis. If the slope at the point were negative, we would know that the function outputs were decreasing as we moved from left to right.



slope_at_point(-3)


# We can visualize this change in slope as we move across the x axis.



points = np.asarray(range(-6, 6))
len(points)




fig, axes = plt.subplots(nrows=3, ncols=4, **{'figsize':(14, 8)})
points = np.asarray(range(-6, 6))
x = np.linspace(-10, 10, 100)
y_nl = non_linear(x)

for point, ax in zip(points, axes.flatten()):
    
    m = slope_at_point(point)
    y = linear(x, m=m)
    y_shift = non_linear(point)

    ax.plot(x, y_nl, linestyle='--') # original non-linear function
    ax.plot(x+point, y+y_shift, linestyle='--')  # Tangent line, i.e. slope at point
    ax.scatter(point, y_shift, s=30, c='black')   # point
    formatter(ax=ax, title='slope at point {}'.format(point));
    ax.set_xlim((-10, 10)); ax.set_ylim((-3, 40));
    plt.tight_layout()

# Read from top left to bottom right


# From this, we can develop an understanding of the algorithms that have been discovered which allow us to automatically optmize functions. Lets try to write a simple one ourselves!



def function_to_optimize(x):
    return 3*(x**2)

def derivative_of_function(x):
    return 6*x

minimum_acceptable_threshold = 0.01
x_point = 10.
max_iterations = 30000
scaling_coef = 0.0001

iterations = 0
slope = derivative_of_function(x_point)

while abs(slope) > minimum_acceptable_threshold:
    
    if slope < 0.: #if slope is negative
        x_point += derivative_of_function(x_point) * scaling_coef
    elif slope > 0: # if slope is positive
        x_point -= derivative_of_function(x_point) * scaling_coef

    iterations += 1
    if iterations > max_iterations:
        break
    slope = derivative_of_function(x_point)

    
print("Iterations until optimized: {}".format(iterations))
print("Minimum x: {}".format(x_point))
print("Slope at minimum : {}".format(derivative_of_function(x_point)))


# ## The gradient

# For functions that have a single input (a scalar) that maps to a single output, the derivative gives us a single value that corresponds to the slope of the function at the given point. When we deal with functions that take multiple inputs (i.e. a multidimensional array of inputs), we encounter the notion of a **partial derivative**. The partial derivative is simply the derivative of a part of the function that corresponds to one of the inputs. So for example, if we have a vector...
# 
# $$[x=3, y=4, z=5]$$
# 
# And we pass this to a function...
# 
# $$f(x, y, z) = 2x^2 + 10y^3 - 7z^5$$
# 
# Then the partial derivatives of this **multivariable function** corresonpond to the inputs x, y and z. So if we wish to optimize this function using derivatives, we can take the partial derivatives by assessing the derivative of each variable, one at a time. 
# 
# The vector of partial derivatives that we produce by doing is referred to as the **gradient** of the function. We can optimize multivariable functions by computing the gradient similar to how we optimized the function using the derivitive in the previous example.

# #### Example: Minimizing a multivariable linear function by descending the gradient

# Lets say we have a multivariable linear function...
# 
# $$f(x) = \frac{1}{2}(Ax − b)^2$$
# 
# ...where $x$ is multivariable vector of inputs, $A$ is a matrix of linear functions, and b is a bias vector. We can think of this matrix as a multivariable function since there are multiple inputs mapping to specific bits of the function, which are then combined together in a predefined way. We use the concept of the gradient to minimize this function by doing, in essence what we did above with our toy minimization algorithm.
# 
# Lets define these variables so we have a concrete understanding of what we mean by the math notation:



x = np.random.randint(0, 10, size=(5, 1))  # We chose to make this function take 5 variables as input
A = np.random.randint(0, 25, size=(4, 5))  # We define the matrix A to be compatible with x
b = np.random.randint(0, 10, size=(4, 1))  # We define b to be compatible with x


# Our input vector could be any shape (depending on the data we are modeling on), but we choose 5 inputs here arbitrarily. The shape of matrix $A$ is chosen to be compatible with the input, but the coefficients of $A$ (i.e. the values of $A$) are chosen at random. If this were a real world example, we might choose more meaningful values for $A$. In this example, we aren't trying to change the values of $A$, we're trying to change the values of $x$.



A




x




b


# Lets compute the output of our function given our randomly chosen starting input values.



0.5 * (A.dot(x)) - b


# Cool, it works!
# 
# Now, to find the vector $x$ that minimizes the ouput of this function, we compute the gradient, and then step the input vector in the appropriate direction. So the first thing to do here is to compute the gradient by finding the function that outputs the partial derivatives of the original function.
# 
# Deriving gradients can seem like a bit of a leap because at this point we probably aren't too comfortable with manipulating the notation. Thats okay! We'll show how this is derived just so we aren't causing any unexplainable leaps in information. But don't worry too much about this - its meerly here to show that it can be done.
# 
# To compute the gradient, we can first expand the function so that we can break it in to individual terms. Remember, the gradient is just the partial derivatives with respect to the input vector $x$, and when we find partial derivatives, we hold all other terms constant. And recall that the derivatives of constants evaluate to zero since they don't change over time/inputs. Also recall that $b$ is a vector of constants.

# $$f(x) = \frac{1}{2}||Ax − b||^2$$
# 
# To compute the gradient, we can use the chain rule to quickly derive the derivative of this function, which is summarized as: Derviative of outside multiplied by derivative of inside.
# 
# $$\bigtriangledown f(x)= \frac{dy}{du}\frac{du}{dx}$$
# 
# $$ =  \frac{2}{2}||Ax - b||^1 *  \frac{du}{dx}$$
# 
# $$ = (Ax - b) * A $$ 

# The reason this works out is because the matrix A is a matrix of coefficients that corespond to a linear function. The derivative of a linear function is the value of the coefficients.
# 
# https://math.stackexchange.com/questions/1990066/how-does-ax-btax-b-reduce-to-xtatax-%E2%88%92-2btax-btb/1990086

# With the form of the gradient identified, we can create a super simple algorithm to minimize the input vector $x$!



eps = 0.0000001
for i in range(2000000):
    grad = np.matmul(A.T, np.matmul(A, x)) * eps
    x = x - grad
print(x)


# # Network Error Minimization using the gradient

# So far, we've talked about minimizing a function using the gradient. But, as we've seen, when we traing a neural network we aren't trying to minimize the output by modifying the input. We are trying to minimize the difference between a prediction and some label that we provide.



## This code will not run
try:
    epochs = 50000
    lr = 0.00001
    batch_size = 5
    input_size = 13
    hidden_size = 10
    samples_seen = 0

    # f(x)_1
    parameteres_hidden_matrix = np.random.normal(loc=0.0, scale=0.01, size=(13, 10))

    # f(x)_2
    parameters_output_matrix = np.random.normal(loc=0.0, scale=0.01, size=(10, 1))  # a single output!

    # Train the network
    for each_epoch in range(epochs):
        batched_data = data_batcher(boston.data, boston.target, batch_size=batch_size)
        epoch_loss = 0

        for batch_input, batch_target in batched_data:

            samples_seen += 1 * batch_size

            # Forward Pass
            hidden_layer_out = np.dot(batch_input, parameteres_hidden_matrix) # No activation function
            pred = np.dot(hidden_layer_out, parameters_output_matrix) # No activation function

            # loss
            epoch_loss += loss_function(pred, batch_target)

            # backprop
            l2_gradient_vector = loss_function_deriv(pred, batch_target) # No derivative
            l1_gradient_vector = np.dot(l2_gradient_vector, parameters_output_matrix.T)  # No derivative since no nonlinear activation!

            # gradient descent
            parameters_output_matrix += np.dot(hidden_layer_out.T, l2_gradient_vector) * lr
            parameteres_hidden_matrix += np.dot(batch_input.T, l1_gradient_vector) * lr
except:
    pass


# 
# 
# $$f_1(x)$$
# 
# Since the outputs of the first layer are fed as inputs to the second layer, we can write the second layer as 
# 
# $$f_2(f_1(x))$$
# 
# If we wish to add more layers, we continue to nest the functions. For example...
# 
# $$f_4(f_3(f_2(f_1(x))))$$

# ### Backpropogation

# When we talk about finding the gradient, we mean we want the vector of partial derivatives for each output - and we use the weight matrix to compute this since this is the thing that represents our function.
# 
# The transfer of error between each layer is called **backpropogation** and it is made possible by the chain rule, which states that the derivative of two nested functions (or in other words, a function that can be interpreted as nested) is the product of the derivitive of the outer function when the inner funciton is substituted for a single variable and the derivative of the inner function. As it turns out, you can actually apply this rule to functions that are nested 2, 3, 4 or more times... such as a neural network.

# ### Gradient Descent

# If you think of the neural network as an optimization problem, then it has some plot like we've been seeing that is neither 2d nor 3d, but something higher dimensional. You can think of the output $pred$ as a point on the surface of this plot, and the labels we've provided as the points that define the surface and its contours. When we use the gradient, we find directions on this surface that point us towrds parameter values that minimize the error.
# 
# For example the plot below shows the 3d surfacefor of a function that maps two variables $x$ and $y$ to a third dependant variable $z$. 



def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 90, cmap='binary')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z');
ax.view_init(55, 35)


# The dimensions of this plot in for a parameterized function such as a neural network are governed by the number of parameteres in the function. In other words, each parameter is another axis. This is why the **partial derivative** is so important - we need the direction to travel for each axis. 
# 
# Since we don't really know what the surface of the target looks like, it stands the reason that we should try to make as much use as we can of the gradient information that can derive from the parameterized function we mean to optimize.
# 
# When we compute the matrix of partial derivaties, we call this matrix a **Jacobian** matrix. Gradient descent makes use of the Jacobian and it tell us the directions in which we should step. It doesn't tell us how much we should step or if an directions are more important than others. For example, imagine that we need to descend the surface of a function that looks like $f(x) = x^2 - y^2$



def f(x, y):
    return x**2 - y**2

x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='winter_r')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z');
ax.view_init(25, 55)

# ![coolfunction](https://github.com/paulegradie/ML_bookclub/blob/master/3_function_optimization/images/cool_function.png "x^2-y^2")


# Gradient descent will give us the directions we need to descend, however in order to gain information about the importance of each step, we need the second dirivatives of our function. this is the derivative of the derivative and the matrix of second derivatives (derived from the jacobian) is referred to as the Hessian.
# 
# We don't really need to go beyond this point, but its useful to know that optimization strategies that make use of the Hessian take advantage of directional importance which can be useful for speeding up optimization and avoiding critical points such as saddle points.
# 
# Saddle points are those points where the derivative is zero, but there is still further to descend without being at a local minium.



# optimization in 2d
def convex(x):
    return  x**2

def non_convex(x):  
    return (x**4) + (x**3) - (6*x**2)

def curve_saddle(x):  
    return (x**4) + (3*x**3) - (x**2)


x = np.linspace(-10, 10, 200)
funcs = [convex, non_convex, curve_saddle]
titles = ['convex', 'non-convex', 'saddle point']

fig, axes = plt.subplots(nrows=1, ncols=3, **{'figsize':(16, 5)})
for title, func, ax in zip(titles, funcs, axes.flatten()):
    y = func(x)
    ax.plot(x, y, linestyle='--'); ax.scatter(0, 0, s=50)
    formatter(ax=ax, title=title)
    ax.set_ylim((-15, 15)); ax.set_xlim((-5, 5))


# To mitigate the problem of local minima, we use various techniques while training such as regularization and certain aproaches to initialization of parameter (weight) matrices. Typically, for feed forward networks, you'll find that peopel typically intiialize the weights drawing values from a normal distribution with mean of zero and a standard deviation of 0.01.



np.random.normal(0.0, 0.01, (5, 4))


# # Bonus content: Deriving the gradient of sum of squares

# 
# The leading $\frac{1}{2}$ is simply a scaling factor, so we can ignore this.
# 
# $ = (Ax − b)^2$
# 
# $ = (Ax-b)^\top(Ax-b)$
# 
# $ = ((Ax)^\top-b^\top)(Ax-b)$
# 
# $ = (A^\top x^\top - b^\top)(Ax-b)$
# 
# $ = (x^\top A^\top A - x) - (b^\top Ax) - (x^\top A^\top b) + (b^\top b)$
# 
# 
# From here we isolate the terms.
# 
# $f(x)_1 = x^\top A^\top A - x$
# 
# $f(x)_2 = b^\top Ax  \leftarrow (scalar)$
# 
# $f(x)_3 = x^\top A^\top b$
# 
# $f(x)_4 = b^\top b  \leftarrow (scalar)$
# 
# Now we find the partial derivatives. In other words, we find the derivative for each of these functions by setting each other term to zero.
# 
# $$gradient = [\bigtriangledown f(x)]_p = \frac{\partial f}{\partial x_p}$$
# 
# So the gradient in this case is defined as the partial derivitives of the function f with respect to the input variables $x$. So anytime there is no x in the function, the gradient will be zero. 
# 
# Thus...
# 
# 
# $$ \bigtriangledown f(x)_1 = \frac{\partial f_1}{\partial x_p} = A^\top Ax $$
# 
# $$ \bigtriangledown f(x)_2 = \frac{\partial f_2}{\partial x_p} = 0         $$
# 
# $$ \bigtriangledown f(x)_3 = \frac{\partial f_3}{\partial x_p} = A^\top b  $$ 
# 
# $$ \bigtriangledown f(x)_4 = \frac{\partial f_4}{\partial x_p} = 0         $$
# 
# Recombining the partial derivatives yeilds the function:
# 
# $$\bigtriangledown f(x) =  A^\top Ax - A^\top b$$
# 
# So to summarize:
# $$\bigtriangledown_x  f(\frac{1}{2}||Ax − b||^2) = A^\top Ax-A^\top b$$

