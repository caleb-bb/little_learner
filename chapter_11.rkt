#lang racket
(require malt)

;; Let a layer be a set of neurons that all operate on one tensor.
;; We can implement this using a layer function. A layer function has this form:
;;  (lambda (t) (lambda (theta) [body what produces a tensor]))
;;  Notice that this is also target function. Let's go over what a target
;; function is again. We have a set of inputs (arguments or "x's") and a set of
;; outputs (y's). We want to create a function that maps the given inputs to the
;; given outputs. The function we want to create is our TARGET function. It's the
;; target because it's the thing we are trying to achieve, the thing we're aiming
;; at. So, where does the target function fit into all the shit we've defined so
;; far?
;;
;; Simple. The target function begins its life as a function expecting
;; arguments, which returns a function expecting parameters. The target function is
;; continually revised over the course of our learning cycles (in the case of this
;; kind of learning, gradient descent) until we hit the TARGET and get a good
;; approximation.
;;
;; The target function is passed as an argument to the loss function. Given what
;; the loss function does, this makes perfect sense: in order to revise our target
;; function, we need to know how far we are from the desired output and which way
;; to go. The loss function gives us that. So the target function ought ot be
;; passed to the loss function, since the loss function knows what to do with the
;; target function.
;;
;; Recall that the layer of neurons is the thing we want to train to produce a
;; certain result. In that sense, the neuron layer IS the function we're adjusting,
;; by adjusting all of its weights. Accordingly, a layer function is a target
;; function.

;; So, let's define our first layer function as an example. First of all, let
;; there be four tensors of rank 1 (that is, four vectors) representing four
;; weights. Notice that each individual weight, in this case, is a vector.
;;
;; Let each weight have shape (list 7)
;;
;; Let there also be four biases, represented as a single tensor of rank 1
;; having the shape (list 4).
;;
;; It follows that w has shape (list 4 7)
;;
;; Therefore, theta looks like this: (list w b)
;; == (list (list 4) (list 4))
;; == (list
;;      (list (tensor 7) (tensor 7) (tensor 7) (tensor 7))
;;      (list 4))
;;
;;  We use the layer by feeding the same inputs to all the neurons in that
;; layer, taking the sum of each neuron's output, and then collecting those sums
;; into a tensor.
;;                         (tensor S1 S2 S3 S4 S5 S6 S7)
;;             /--N1---------------/ /  /  /  /  /  /
;;            /---N2----------------/  /  /  /  /  /
;;           /----N3------------------/  /  /  /  /
;;       [I] -----N4--------------------/  /  /  /
;;           \----N5----------------------/  /  /
;;            \---N6------------------------/  /
;;             \--N7--------------------------/

(define linear-1-1 (lambda (t) (lambda (theta) (+ (dot-product (tref theta 0) t) (tref theta 1)))))
(define relu-1-1 (lambda (t) (lambda (theta) (rectify ((linear-1-1 t) theta)))))

(define layer-function-first-pass
  (lambda (t)
    (lambda (theta)
      (let ([w (ref theta 0)]
            [b (ref theta 1)])
        [((tensor ((relu-1-1 t) (list (tref w 0) (ref b 0)))))
         ((tensor ((relu-1-1 t) (list (tref w 1) (ref b 1)))))
         ((tensor ((relu-1-1 t) (list (tref w 2) (ref b 2)))))
         (tensor ((relu-1-1 t) (list (tref w 3) (ref b 3))))]))))

;; Let L be a layer. For our purposes, L is a layer *function*.
;; Let L be a *dense* layer.
;; Let L be composed of neurons, N1...Nm, where each neuron is a function.
;;
;; Given that L is a *dense* *layer* *function* having m neurons, we know that L:
;; 1. Takes a first argument, t, is a tensor of shape (list n)
;;      1.a t is a tensor of rank 1 (that is, a vector)
;; 2. Takes a second argument, theta, contains weights and biases. The weights have shape (list m n) and the biases have shape (list m)
;; 3. Calls each N on t
;; 4. Collects the scalar outputs of each N into a tensor of rank 1 (that is, a vector) having shape (list n).
;;
;; THE LAW OF DENSE LAYERS - SLIGHTLY TARDED VERSION
;; A dense layer function invokes m neurons on an n-element input vector and produces an m element output vector.
;;
;; THE LAW OF DENSE LAYERS - FINAL VERSION
;; A dense layer function invoks m neurons on an n-element input tensor having
;; rank one (that is, a vector) that produces an m-element output tensor having
;; rank one (that is, a vector) in a single invocation of multiplication-2-1
;;
(define dot-product-2-1 (lambda (w t) (sum (*-2-1 w t))))

;; (*-2-1 (tensor (tensor 0.2 0.3) (tensor 0.5 0.7)) (tensor 0.11 0.13))
(*-2-1 (tensor (tensor 2 3) (tensor 5 7)) (tensor 11 13))

;; To reiterate: the dot product is basically the sum of the pairwise product of two tensors
;; For two vectors, this is simply one-to-one multiplication. For higher-ranking
;; tensors, we apply the generalizations/extensions of operations defined earlier
;; in the books.

;; Recall that *-2-1 == (ext2 * 2 1)
;;
;; Recall that
;;  (define ext2
;;      (lambda (f n m)
;;          (lambda (t u)
;;              (cond
;;              [(of-ranks? n t m u (f t u))
;;              [else (desc ext2 f n m) n t m u]]))))
;;
;;  Fundamentally, this function lets us take any given function in two
;; arguments (that operates on tensors), define a pair of ranks for tensors at
;; which that function should be applied, and then descends through the tensors
;; until we get the ranks we want so that we can apply that function. This gives us
;; a higher-order function that returns extended functions.
;;
;; Basically, the two ranks that we pass in tell ext2 when to stop descending and apply the function.
;;
;;
;; 1. (dot-product-2-1
;;      (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1))
;;      (tensor 1.3 0.4 3.3))
;;
;; 2. (sum (*-2-1                                                           eval dot-product-2-1
;;      (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1))
;;      (tensor 1.3 0.4 3.3))))
;;
;; 3. (sum                                                                  eval *-2-1
;;      ((ext2 * 2 1)
;;          (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1))
;;          (tensor 1.3 0.4 3.3)))
;;
;; 4. (sum                                                                 eval ext2
;;      (cond
;;          ([(of-ranks? 2 (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1)) 1 (tensor 1.3 0.4 3.3))]
;;              (* (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1)) (tensor 1.3 0.4 3.3)))
;;          (else ...)))
;;
;; 5. (sum                                                                 eval cond
;;      (*
;;          (tensor
;;              (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1))
;;          (tensor 1.3 0.4 3.3)))
;;
;; 7. (sum                                                                 eval * (remember that * is already extended to rank 1 tensors in malt!)
;;      (tensor
;;          (* (tensor 2.0 1.0 3.1) (tensor 1.3 0.4 3.3))
;;          (* (tensor 3.7 4.0 6.1) (tensor 1.3 0.4 3.3))))
;;
;; 8. (sum                                                                 eval *
;;      (tensor
;;          (tensor 2.6 0.4 10.23)
;;          (tensor 4.8100000000000005 1.6 20.13)))
;;
;; 9. (tensor 13.23 26.54)                                                 eval sum

;; REFRESHER ON FUNCTION EXTENSION
;; When we talk about extending a function, what do we mean?
;; Call an operator binary if it's a relation on a set taking two arguments and
;; returning a single result, viz. maps pairs onto singles within a set.
;; Let S be the set of all binary operators defined on the set of scalars.
;; Let T be the set of all binary operators defined on the set of tensors.
;; When we "extend" a function, we are mapping S to T. The specific mapping
;; changes based on what we want.
;;
;; Generally speaking, if the two tensors we input to the operation have the
;; same rank, we want pairwise appiclation of the binary scalary operator to all
;; scalars in both tensors

(define linear (lambda (t) (lambda (theta) (+ (dot-product-2-1 (ref theta 0) t) (ref theta 1)))))
(define relu (lambda (t) (lambda (theta) (rectify ((linear t) theta)))))

;; If a layer of neurons has m neurons and the input length is n, then the theta
;; provided should be such that (ref theta 0) has the shape (list m n) and (ref
;; theta 1) has the shape (list m)
;;
;; Every artificial neuron has a one-dimensional tensor of weights and a scalar
;; for a bias. Therefore, the output tensor for a given layer always has shape
;; (list n) where n is the length of the bias vector, which is also the first
;; number in the shape of the weight tensor.
;;
;; Moreover, the number of arguments-of-line, that is, the number of inputs, is
;; also equal to n.
;;
;; To recap: the number of inputs (or x's), the first number in the shape of the
;; weight tensor, and the length of the bias vector, are all the same nnumber, and
;; we use that to design our networks.
;;
;; The length of the output vector, however, is always equal to the number of
;; neurons.
;;
;; One way to look at it: each neuron has n weights, where n is the number of
;; inputs. There are k neurons. Therefore, the input tensor must have shape (list k
;; n), meaning it's basically k copies of the input vector. Remember that neurons
;; always output a weighted SUM, meaning they output a scalar. Therefore, the
;; output of a given layer must be a vector having length n.
;;
;; Note that the bias vector also has length n.

;; This function is kinda useless because it is literally just relu wrapped in plastic
;; But it demonstrates a thing
(define 1-relu (lambda (t) (lambda (theta) ((relu t) theta))))

;; This looks horrendously complicated but it actually makes sense. Consider:
;; 1. Each layer is made of neurons.
;; 2. Each neuron has a tensor of weights
;; 3. Theta contains the list of weights for the whole network
;; 4. Therefore, the first number in shape(theta) is the number of layers
;; 5. Which means that theta, at the highest level, is a list of tensors, each
;; one intended for a specific layer.
(define 2-relu (lambda (t) (lambda (theta) ((relu ((relu t) theta)) refr theta 2))))
(define 3-relu (lambda (t) (lambda (theta) ((2-relu ((relu t) theta)) (refr theta 2)))))
(define k-relu-1
  (lambda (k t theta)
    (cond
      [(zero? k) t]
      [else (k-relu (sub1 k) ((relu t) theta) (refr theta 2))])))

(define k-relu-2
  (lambda (k)
    (lambda (t theta)
      (cond
        [(zero? k) t]
        [else
         ((k-relu (sub1 k)))
         ((relu t) theta)
         (refr theta 2)]))))

;; The difference here is very subtle. Basically, at each recursive step, we
;; evaluate k-relu with one argument, amd then feed the evaluation of the base case
;; and the rest of theta to it.
;;

(define k-relu
  (lambda (k)
    (lambda (t)
      (lambda (theta)
        (cond
          [(zero? k) t]
          [else (((k-relu (sub1 k)) ((relu t) theta) (refr theta 2)))])))))

;; For a brief sketch of how this plays out:
;; (k-relu k t theta)
;; => ((k-relu k-1) ((relu t) theta) (refr theta 2))
;; == ((k-relu k-1) (result of relu 1) (theta 2..n)
;; => ((k-relu k-2) (result of relu 2) (theta 4..n))
;; => ((k-relu k-3) (result of relu 3) (theta 6..n))
;; => ((k-relu k-4) (result of relu 4) (theta 8..n))
;; => ((k-relu k-5) (result of relu 5) (theta 10..n))
;;
;; So, what this function does:
;; 1. Takes the number of layers, as k
;; 2. Takes the input tensor, t
;; 3. Takes theta, which is a tensor. Each 2-entry "slice" of theta is a weight tensor and a bias vector, respectively.
;; 4. Finds the output of the first layer, using the input tenso entries 0 and 1 from theta. This ouput which becomes the next t (input tensor).
;; 5. Slices off the first two entries of theta, so that the next two are weights and biases for the next layer
;; 6. Calls itself with the output of the first layer as t and a new theta whose first two entrys
;;
;; Let there be a 3-layer network where the input is a vector of shape (list 32). Let the input layers have width 64, 45, and 26.
;;
;; Network function: k-relu 3
;; Length of theta: 2 * number of layers = 6
;; And the shapes of thetas 0, 2, and 4?
;; Their shapes are (list 32 64), (list 32 45), and (list 32 36), respectively
;; The shapes of the bias vectors are (64, 45, and 36)
;; So the shape list for the network is
;; (list
;; (list 64 32)
;; (list 64)
;; (list 45 64)
;; (list 45)
;; (list 26 45)
;; (list 26))
