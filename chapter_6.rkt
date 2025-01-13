#lang racket
(require malt)
;; What is batching?
;; Batching is, fundamentally, a way of calculating loss without traversing the
;; entire data set every time. Instead, we randomly sample the data set and
;; calculate loss using only the sample. If we do this enough times, we can get a
;; good-enough approximation to revise our weights.

;; A loss function can be thought of as a function that requires three arguments:
;; 1. A set of parameters (inputs)
;; 2. A set of expected outputs for those parameters (training set?)
;; 3. A function to generate predictions with (predictive function)
;; 4. A set of arguments to give to the predictive function (weights and biases)
;;
;; Note that this is not STRICTLY true. Generally, the predictive function is a
;; part of the model itself, not an argument to another function. However, if we
;; see this in terms of currying, we get a more generalized understanding. We
;; realize, for example, that one could easily define a program to generate
;; different models - and see HOW this could be done.
;;
;; Additionally, a loss function needs some way to measure the loss. We can
;; think of that measuring as the core of the loss function. There are many
;; different ways to measure loss. For example, we can take the square of each
;; scalar in the whole tensor, and then take the sum of all the rank-1 tensors.
;; Notice that this reduces the rank of the overall tensor by 1.
;; Another way to do it: take the ABSOLUTE VALUE of every scalar, and then do
;; the same process. Both strategies want to make sure all the values are positive.
;;
;; There are many other ways as well, of course. The exact nature of the loss
;; function is implementation-specific.
;;
;; If we evaluate our loss function with only the third argument, then we get an
;; "expectant function". This function EXPECTs a data set.
;;
;; That data set include: a set of inputs, a set of expected outputs for those
;; inputs, and a set of weights and biases to give to the predictive function.
;;
;; If we evaluate the expectant function with inputs and expected outputs, but
;; no weights/biases, then we get an objective function. The objective function
;; only wants weights and biases, at which point it returns loss.
;;
;; "Objective function"
;;
;; So we have:
;; (loss_function)
;; (loss_function predictive_function)
;;      => (expectant_function)
;; (expectant_function inputs_and_outputs) == ((loss_function predictive_function) inputs_and_outputs)
;;      => (objective_function)
;;    (objective_function weights_and_biases)
;; == ((expectant_function inputs_and_outputs) weights_and_biases)
;; == (((loss_function predictive_function) inputs_and_outputs) weights_and_biases)
;; == loss
;;
;; If you give a loss function a predictive function, it returns an expectant function.
;; If you give an expectant function a set of inputs and outputs, it returns an objective function.
;; If you give an objective function a set of weights and biases, it returns a loss.
;;
;; A gradient is the change loss divided by change in weights and biases.
;; In other words, THE GRADIENT IS THE DERIVATIVE OF THE OBJECTIVE FUNCTION.
;;
;; Gradient descent, then, is very simple:
;; 1. Feed a set of weights and biases to an objective function
;; 2. Take the loss and use it to compute the derivative of the objective function FOR EACH INPUT.
;; 3. Take that derivatives (the gradient) and use them to revise the weights to shrink the loss
;; 4. Return to step 1 until either you run out of iterations or you're
;;    satisfied with how close you are to the desired outputs.
;;
;; Step 2 needs some justification. Each separate input is NOT a change in a
;; variable. It is better to think of the separate inputs as their own variables.
;; Mathematically, we can take the derivative of the objective function "with
;; respect to" any variable. So if there are i weights (meaning i inputs, since
;; there's an input for each weight) then there are i different derivatives of the
;; function. That means that the gradient is a set of derivatives with i members.
;;
;; Step 3 is implemented by multiplying the gradient by a fixed "learning rate".
;; Since the gradient is a vector containing all partial derivatives of the loss
;; function, it follows that multiplying it by the learning rate yields a vector of
;; small values. This vector can be subtracted from the weight vector to give us
;; our next iteration. Gradients always point in the direction of steepest ASCent,
;; so we want to subtract the product of each value in the gradient and the
;; learning rate from the original weight to shrink the overall loss.
;;
;; We have "hyperparameters", which are sort of meta to the whole process. In
;; this case, the hyperparameters are the learning rate (sometimes called "alpha"),
;; and the number of revisions. There are othe rpossible hyperparameters,
;; depending on implementation. For example, batch size, which specifies how big
;; each randomly-sampled batch shold be. Another one is momentum, which lets us
;; adjust the weights by a different amount each iteration to reduce the number of
;; overall iterations needed.

;; THE RULE OF BATCHES
;; A batch of indices consists of random natural numbers < the input tensor's length.
;; We need a function for sampling. It take two arguments. The first, n, is the
;; number of points in the data set. The second, s, is the size of the sample set.
;; Both are natural numbers greater than one, and s is less than or equal to n.
;;
;; The sampling function returns a list with s members, each being a randomly
;; chosen natural number < n. It is defined thus:

(declare-hyper batch-size)
(declare-hyper revs)
(declare-hyper alpha)

(define samples
  (lambda (n s)
    (sampled n s (list))))

(define sampled
  (lambda (n i a)
    (cond
      ((zero? i) a)
      (else
       (sampled n (sub1 i)
       (cons (random n) a))))))

;; This is similar to Enum.reduce in Elixir.

(define a-tensor (tensor 5.0 2.8 4.2 2.3 7.4 1.7 8.1))


(define sampling-obj
  (lambda (expectant xs ys)
    (let ((n (tlen xs)))
          (lambda (theta)
            (let ((b (samples n batch-size)))
              ((expectant (trefs xs b) (trefs ys b)) theta))))))

(define line-xs (tensor 2.0 1.0 4.0 3.0))
(define line-ys (tensor 1.8 1.2 4.2 3.3))

;; (with-hypers
;;   ((revs 1000)
;;    (alpha 0.01)
;;    (batch-size 4))
;;     (gradient-descent
;;         (sampling-obj
;;          (l2-loss line) line-xs line-ys)
;;         (list 0.0 0.0)))

;; THE LAW OF BATCH SIZES
;; Each revision in stochastic gradient descent uses only a batch of size
;; batch-size from the data set and the ranks of the tensors in the batch are the
;; same as the ranks of the tensors in the data set.

(define plane-xs
  (tensor
   (tensor 1.0 2.05)
   (tensor 1.0 3.0)
   (tensor 2.0 2.0)
   (tensor 2.0 3.9)
   (tensor 3.0 6.13)
   (tensor 4.0 8.09)))

(define plane-ys (tensor 13.99 15.99 18.0 22.4 30.2 37.94))

(define gradient-descent
  (lambda (obj theta)
    (let ((f (lambda (big_theta)
               (map (lambda (p g)
                      (- p (* alpha g)))
                         big_theta
                         (gradient-of obj big_theta)))))
          (revise f revs theta))))

(with-hypers
  ((revs 15000)
   (alpha 0.001)
   (batch-size 4))
  (gradient-descent
   (sampling-obj
    (l2-loss plane) plane-xs plane-ys)
   (list (tensor 0.0 0.0) 0.0)))
