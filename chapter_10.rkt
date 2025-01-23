#lang racket
(require malt)

;; rectify takes a scalar and returns 0 if the scalar is negative. Otherwise, it returns the scalar.
(define rectify-0
  (lambda (s)
    (cond
      [(< s 0.0) 0.0]
      [else s])))

; This extends rectify to tensors
(define rectify (ext1 rectify-0 0))

;; These kinds of small, nonlinear functions, like rectify, are known as deciders,
;; or activation functions. They make a decision about the arguments and transfer
;; that decision down the pipeline.

;; the "superscript" of 1-1 here reminds us that this function expects t and theta 0 to be tensors
;; It takes the tensor, t, and the tensor at theta_0, and
(define linear-1-1 (lambda (t) (lambda (theta) (+ (dot-product (tref theta 0) t) (tref theta 1)))))

;; 'relu' stands for "rectifying linear unit". It makes a WEIGHTED DECISION about its argument tensor t
(define relu-1-1 (lambda (t) (lambda (theta) (rectify ((linear-1-1 t) theta)))))

;; When we look at theta_0, we see a tensor full of weights. These are the
;; familiar weights from artificial neurons. theta_1, by contrast, is a tensor of
;; biases. So when we 'inflate' a theta, we're taking a tensor of weights and
;; adding our biases to them; the process of updating the weights is the update
;; function fed into revise; and the process of deflation just gives us the updated
;; weights, which will be inflated again.

;; 1. ((relu-1-1 (tensor 2.0 1.0 3.0))
;; (list (tensor 7.1 4.3 -6.4) 0.6))
;;
;; 2. (rectify
;; (+
;; (dot-product (tensor 7.1 4.3 -6.4) (tensor 2.0 1.0 3.0))
;; 0.6))

;; THE RULE OF ARTIFICIAL NEURONS
;; An artificial neuron is a parameterized linear function composed with a nonlinear decider functionA
;;
;; 'parameterized' here means "accepting parameters". Recall that an ARGUMENT is
;; what we frequently call an input, e.g. the x-coordinate to a linear function,
;; and a PARAMETER is all the stuff coming after the argument.
;;
;; 'linear' here means "using only addition and scaling". scaling just means
;; multiplying an argument by a scalar.
;;
;; So, to take apart our rule of artificial neurons, an artificial neuron is
;; n(arg) = l(d(arg)). We take an argument and send it to a decider function, which
;; makes a small decision and passes that to a linear function. That's it
;;
;; Let us begin by assuming that, for any given x, thhere exists a theta such that ((relu-1-1 (tensor x)) theta)
;; And let's try to find the y for x = 0.5
;;
;; 1. ((relu-1-1 (tennsor 0.5)) (list (tensor 1.0) -1.0))                   Evaluate relu-1-1
;; 2. (rectify
;;  (+
;;  (dot-product (tensor 1.0) (tensor 0.5))
;;      -1.0))
;;
;; 3.     (rectify                                                          Evaluate dot-product
;;          (+
;;              (sum
;;                  (* (tensor 1.0) (tensor 0.5)))
;;                      -1.0))
;; 4. (rectify -0.5)
;; 5. 0.0

(define half-strip
  (lambda (x theta)
    (- ((relu-1-1 (tensor x)) (list (tref theta 0) (tref theta 1))) ((relu-1-1 (tensor x))))))

(define full-strip
  (lambda (x theta)
    (- (half-strip x (list (trefs theta 0 1 2))) (half-strip x (list (trefs theta 3 4 5))))))

;; Notice that relu-1-1 is a artifical neuron. That's because linear-1-1 is a
;; parameterized linear function (takess a slope and a bias) composed with a
;; nonlinear decider function (rectify). Accordingly, the number of relu-1-1
;; invocations required to implement a certain function tells us how many
;; artificial neurons we need to model that thaaang.

;; Notice that half-strip and full-strip are both, technically speaking, neural
;; networks. They're neural networks because they link more than one neuron
;; together. The whole point of this chapter is to understand what an artificial
;; neuron is. The mathematical definition is that it's a linear parametric function
;; composed with a nonlinear decider function. A more accessible definition is to
;; say that an artificial neuron is a system that will find a linear function that
;; loosely fits some set of points. You've got your y = mx +b, or, in our case, y =
;; wx + b. The artificial neuron's job is to take an argument (x) and an output (y)
;; and find the w and b that gives us that y. Of course, x and y are not necesarily
;; single values here; we speak more accurately in terms of SETS of x's and SETS of
;; y's.
