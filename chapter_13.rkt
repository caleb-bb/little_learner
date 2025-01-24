#lang racket
(require malt)
(require malt/examples/iris)

;; The input of a neural network is always a tensor. The output, likewise, is
;; always a tensor. To use a neural net for anything besides numbers, we have to
;; find ways of turning things like words and sounds and images into tensors and
;; back again.
;;
;; We call this "encoding" and "decoding".
;;
;; One of the simpler methods of encoding is called "one-hot". In one-hot, you
;; figure out how many classes of things you have to classify stuff into. The
;; number of classes is equal to the length of a binary input tensor. Each class as
;; an index asociated therewith. The value at that index is set to 1, and all other
;; values are set to 0.
;;
;; So, if we have three classes, Red, Blue, and Yellow, we could represent them like so:
;;
;; Red      => [1.0 0.0 0.0]
;; Blue     => [0.0 1.0 0.0]
;; Yellow   => [0.0 0.0 1.0]
;;
;; These three floats are understood as percentages reflecting our degree of
;; confidence that something falls into that class. If they're floats, it's
;; one-hot-LIKE. if they're always 1 or 0, it's one-hot.
;;
;; We could also do dogs and cats:
;;
;; Dog      => [1.0 0.0]
;; Cat      => [0.0 1.0]
;;
;; Note that 100% certainty that a thing is a dog is 100% certainty that is NOT
;; a cat, which is to say, 0% certainty that it's a cat
;;
;; If we're using one-hot, then the output layer should have a width equal to
;; the length of the classification vector.
;;
;; For one-hot-like encoding, the predicted class is max(classification_vector)
;;
;; Generally speaking, layers closer to the output are narrower than layers
;; closer to the input. This is because layers closer to the input are going to
;; learn the more primitive features (smaller-scale) of the data, while
;; higher-level features, more abstract ones based on the input data, are learned
;; by neurons further down the stack.

(define dense-block (lambda (n m) (block relu (list (list m n) (list m)))))
(define iris-network (stack-blocks (list (dense-block 4 6) (dense-block 6 3))))

;; Initializing a dataset with weights all the same can, for some certain
;; networks, lead to deterministic behavior where every neuron makes the same
;; decision over and over and every scalar in the output tensor is the same every
;; time.
;;
;; This is a problem, because in order to detect different features of a data
;; set, each neuron must have a unique value. How do? Answer: randomize.
;;
;; Also, a scalar in an input tensor will make its way through every layer. That
;; means it gets multiplied by many many weights until it hits the output tensor.
;; If all those weights are large numbers, then our output will be a set of big
;; fuckoff scalars we can't use. This is called EXPLODING. On the other hand, if
;; the the weights are too small, then the output peters out very close to 0,
;; wwhich is called VANISHING.
;;
;; So, let there be a function called random-tensor. random-tensor takes three
;; arguments: c, v, and s.
;; Those three arguments are a central value, a variance, and a shape.
;;
;; The best value for variance is (/ 2 n) where n is the width of the input
;; layer. This is specific to networks that use rectify as a decider function and
;; is known as He initialization.
;;
;; THE RULE OF LAYER INITIALIZATION
;; The bias vector of a layer is initialized to contain only 0.0.
;; The weight tensor-2 of a lyer is initialized to random scalars with a central
;; value of 0.0 and a variance of 2/n. where n is the length of the input to the
;; layer.

(define init-shape
  (lambda (s)
    (cond
      [(= (tensor s) 1) (zero-tensor s)]
      [(= (tensor s) 2)]
      [random-tensor
       0.0
       (/ 2 (tref s 1))
       s])))
(define init-theta (lambda (shapes) (map init-shape shapes)))

(define iris-classifier (block-fn iris-network))
(define iris-theta-shapes (block-ls iris-network))

(define iris-theta
  (with-hypers ((revs 2000) (alpha 0.0002) (batch-size 9))
               (naked-gradient-descent
                (sampling-obj (l2-loss iris-classifier) iris-train-xs iris-train-ys)
                (init-theta iris-theta-shapes))))

;; imagine an idealized function that produces the correct y for every x in a
;; data set, AND the correct y for any x NOT in the data set. This function is
;; idealize because we assume its existence but do not have any evidence except
;; that the model gets the right predictions for the ys in the training set
;;
;; We say that a model approximates this idealized function. A model, remember,
;; is an algorithm plus a set of weights. The idea is to use an algo to
;; jiggle the weights to get the model closer to the idealized function.
;;
;; In our case, the algorithm in question is the target function.

(define model (lambda (target theta) (lambda (t) ((target t) theta))))
(define iris-model (model iris-classifier iris-theta))
